import hashlib
import json
import re
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from dateutil import parser as date_parser
import joblib
import numpy as np
from backend.config import Config
from backend.ml.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

class NewsAnalyzerService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_loaded = False
        self.ensemble = None
        self.calibrated = None
        self.active_model_path = None
        self.model_error = None
        self._load_local_model()

    def _load_local_model(self):
        """
        Tenta carregar o modelo treinado pelo usuário a partir da pasta models/.
        Aceita tanto pipelines completos (ex: LinearSVC .joblib/.pkl) quanto modelo + vetorizador.
        """
        self.model = None
        self.vectorizer = None
        self.ensemble = None
        self.calibrated = None
        self.active_model_path = None
        self.model_loaded = False
        self.model_error = None
        self.backend = Config.MODEL_BACKEND
        if self.backend == "calibrated":
            try:
                from backend.ml.calibrated import CalibratedClassifier
                self.calibrated = CalibratedClassifier(Config.CLASSIFIER_PATH)
                self.active_model_path = str(Config.CLASSIFIER_PATH)
                self.model_loaded = True
                logger.info("Classificador calibrado carregado: %s", self.calibrated.version)
            except Exception as exc:
                self.model_error = str(exc)
                logger.error("Falha ao carregar classificador calibrado: %s", exc)
            return
        if self.backend == "ensemble":
            try:
                if not Config.ENSEMBLE_ENABLED:
                    raise ValueError("Ensemble desabilitado por ENSEMBLE_ENABLED.")
                from backend.ml.ensemble import SoftVotingEnsemble
                self.ensemble = SoftVotingEnsemble.from_manifest(
                    Config.ENSEMBLE_MANIFEST, device=Config.TRANSFORMER_DEVICE,
                )
                self.active_model_path = str(Config.ENSEMBLE_MANIFEST)
                self.model_loaded = True
                logger.info("Ensemble carregado de %s", Config.ENSEMBLE_MANIFEST)
            except Exception as exc:
                self.model_error = str(exc)
                logger.error("Falha ao carregar ensemble: %s", exc)
            return
        if self.backend != "legacy":
            self.model_error = "MODEL_BACKEND deve ser calibrated, legacy ou ensemble."
            return
        model_path = Config.get_model_path()
        vectorizer_path = Config.VECTORIZER_PATH

        if model_path and model_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = joblib.load(model_path)
                logger.info(f"Modelo carregado com sucesso a partir de {model_path}")
                
                if vectorizer_path.exists():
                    self.vectorizer = joblib.load(vectorizer_path)
                    logger.info(f"Vetorizador carregado com sucesso a partir de {vectorizer_path}")
                else:
                    self.vectorizer = None
                    
                self.model_loaded = True
                self.active_model_path = str(model_path)
            except Exception as e:
                logger.error(f"Erro ao carregar modelo de {model_path}: {e}")
                self.model = None
                self.vectorizer = None
                self.model_loaded = False
                self.model_error = str(e)
        else:
            self.model_loaded = False

    def is_model_loaded(self) -> bool:
        """Verifica se há um modelo de ML próprio carregado."""
        return self.model_loaded

    def model_status(self) -> dict:
        semantic = getattr(self.ensemble, "semantic", None)
        return {
            "backend": self.backend,
            "loaded": self.model_loaded,
            "path": self.active_model_path,
            "version": self.calibrated.version if self.calibrated else getattr(self.ensemble, "version", None),
            "sha256": self.calibrated.sha256 if self.calibrated else getattr(self.ensemble, "sha256", None),
            "transformer": {
                "loaded": True, "device": str(semantic.device),
                "max_length": semantic.max_length,
                "source_model": semantic.metadata.get("source_model", semantic.metadata.get("base_model")),
                "task": semantic.metadata.get("task"),
            } if semantic is not None and hasattr(semantic, "metadata") else None,
            "error": self.model_error,
        }

    def probability_thresholds(self):
        if self.calibrated:
            return self.calibrated.manifest["limiares_prob_fake"]
        return getattr(self.ensemble, "thresholds", None)

    def reload_model(self) -> bool:
        """Recarrega os arquivos de modelo se o usuário tiver adicionado recentemente."""
        self._load_local_model()
        return self.model_loaded

    def analyze(self, article_data: dict) -> dict:
        """
        Executa a análise completa da notícia extraída pelo scraper:
        - Avalia os 5 critérios (Evidências, Qualidade das fontes, Corroboração, Contexto, Atualidade)
        - Calcula a pontuação de confiança (0 a 100)
        - Preserva as faixas originais do classificador calibrado.
        - No modo legado/ensemble aplica o limiar binário configurado.
        """
        predictor = self.calibrated or self.ensemble
        prediction_mode = "calibrated" if self.calibrated is not None else ("ensemble" if self.ensemble is not None else "legacy")
        if self.backend in ("calibrated", "ensemble") and (not self.model_loaded or predictor is None):
            raise ModelUnavailableError(f"Modelo selecionado indisponível: {self.model_error or 'sem diagnóstico'}")

        text = article_data.get("text", "")
        title = article_data.get("title", "")
        domain = article_data.get("domain", "")
        author = article_data.get("author", "")
        publish_date = article_data.get("publish_date")
        word_count = article_data.get("word_count", 0)
        links_count = article_data.get("external_links_count", 0)

        # 1. Cálculo individual dos 5 critérios
        evidencias = self._evaluate_evidencias(text)
        qualidade_fontes = self._evaluate_qualidade_fontes(domain, author, links_count, text)
        corroboracao = self._evaluate_corroboracao(title, text)
        contexto = self._evaluate_contexto(title, text, word_count)
        atualidade = self._evaluate_atualidade(publish_date, text)

        # Média ponderada dos critérios da matéria
        criteria_avg = (
            evidencias["score"] * 0.25 +
            qualidade_fontes["score"] * 0.25 +
            corroboracao["score"] * 0.20 +
            contexto["score"] * 0.15 +
            atualidade["score"] * 0.15
        )

        # 2. Avaliação pelo Modelo de ML (LinearSVC ou outro em models/)
        ml_score = None
        ml_details = None
        if predictor is not None or (self.model_loaded and self.model is not None):
            try:
                ml_score, ml_details = self._predict_with_user_model(title, text, predictor=predictor)
            except Exception as e:
                logger.error(f"Erro ao inferir com o modelo do usuário: {e}")
                if prediction_mode != "legacy":
                    raise ModelUnavailableError("Falha na inferência do modelo selecionado") from e
                ml_score = None
                ml_details = None

        # 3. Determinação da pontuação final de confiança (0 a 100)
        if prediction_mode != "legacy" and ml_score is not None:
            # O score dos modelos calibrado/ensemble mantém as heurísticas informativas.
            final_score = ml_score
        elif ml_score is not None:
            # Combinação: 70% predição do modelo treinado (LinearSVC) + 30% métricas dos 5 critérios
            final_score = round((ml_score * 0.70) + (criteria_avg * 0.30), 1)
        else:
            final_score = round(criteria_avg, 1)

        # Garante limite estrito de 0 a 100
        final_score = max(0.0, min(100.0, final_score))

        # 4. Faixas do artefato calibrado ou limiar dos modos anteriores.
        has_bands = ml_details is not None and ml_details.get("faixa") is not None
        if has_bands:
            verdict, verdict_label = {
                "verdadeira": ("confiavel", "Provavelmente verdadeira"),
                "inconclusiva": ("inconclusiva", "Resultado inconclusivo"),
                "falsa": ("desconfiavel", "Possível desinformação"),
            }[ml_details["faixa"]]
        elif final_score >= Config.CONFIDENCE_THRESHOLD:
            verdict = "confiavel"
            verdict_label = "Confiável"
        else:
            verdict = "desconfiavel"
            verdict_label = "Não Confiável / Suspeita de Fake News"

        # 5. Síntese textual da análise
        if has_bands:
            summary = (
                f"O classificador estimou {ml_details['prob_fake'] * 100:.2f}% de probabilidade "
                f"da classe falsa; faixa {ml_details['faixa']}. "
                "A análise considera padrões do texto e não confirma fatos em fontes externas."
            )
        else:
            summary = self._generate_summary(final_score, verdict, evidencias, qualidade_fontes, corroboracao)
        input_hash = hashlib.sha256(json.dumps([title, text], ensure_ascii=False).encode()).hexdigest()

        return {
            "status": "success",
            "url": article_data.get("url"),
            "metadata": {
                "title": title,
                "author": author,
                "publish_date": publish_date,
                "domain": domain,
                "word_count": word_count,
            },
            "confidence_score": final_score,
            "verdict": verdict,
            "verdict_label": verdict_label,
            "threshold": None if has_bands else Config.CONFIDENCE_THRESHOLD,
            "thresholds": ml_details.get("thresholds") if ml_details else None,
            "prob_fake": ml_details.get("prob_fake") if ml_details else None,
            "faixa": ml_details.get("faixa") if ml_details else None,
            "model_version": ml_details.get("versao_modelo") if ml_details else None,
            "model_evaluation": ml_details.get("evaluation") if ml_details else None,
            "score_method": {"calibrated": "calibrated_model", "ensemble": "ensemble", "legacy": "legacy_blend"}[prediction_mode],
            "criteria_affect_score": prediction_mode == "legacy",
            "criteria_method": "heuristic_text_signals",
            "provenance": {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "input_sha256": input_hash,
                "model_sha256": ml_details.get("model_sha256") if ml_details else None,
                "extraction": article_data.get("extraction"),
            },
            "used_custom_ml_model": ml_score is not None,
            "ml_details": ml_details,
            "criteria": {
                "evidencias": evidencias,
                "qualidade_fontes": qualidade_fontes,
                "corroboracao": corroboracao,
                "contexto": contexto,
                "atualidade": atualidade,
            },
            "summary": summary
        }

    def _predict_with_user_model(self, title: str, text: str, predictor=None) -> tuple[float, dict]:
        """
        Executa a predição usando o modelo do usuário (ex: LinearSVC).
        Adapta-se a modelos com decision_function, predict_proba ou predict simples.
        Retorna (ml_confidence_score, ml_details).
        """
        predictor = predictor or self.calibrated or self.ensemble
        if predictor is not None:
            if hasattr(predictor, "predict_document"):
                details = predictor.predict_document(title, text)
            else:
                details = predictor.predict(f"{title} {text}".strip())
            return details["ml_confidence_percent"], details
        if self.backend in ("calibrated", "ensemble"):
            raise ModelUnavailableError("Modelo selecionado indisponível durante a inferência.")
        input_content = f"{title} {text}".strip()

        # Probabilidades nativas têm prioridade, inclusive na Regressão Logística.
        if hasattr(self.model, "predict_proba"):
            from backend.ml.ensemble import LexicalClassifier
            adapter = LexicalClassifier(
                self.model, reliable_class=Config.PROBABILISTIC_RELIABLE_CLASS,
                vectorizer=self.vectorizer,
            )
            probs = adapter.predict_proba([input_content])[0]
            score = float(probs[1] * 100)
            return score, {
                "model_type": "Probabilistic Classifier",
                "predicted_class": int(probs[1] >= 0.5),
                "raw_prediction_label": "Classificada pelo modelo",
                "probabilities": {"fake": float(probs[0]), "confiavel": float(probs[1])},
                "ml_confidence_percent": score,
            }

        # Se houver vetorizador separado
        if self.vectorizer is not None:
            X = self.vectorizer.transform([input_content])
        else:
            # Caso o modelo seja um Pipeline sklearn (ex: best_linearsvc_model.joblib)
            X = [input_content]

        # 1. Caso LinearSVC / SVM (utiliza decision_function)
        if hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(X)[0])
            
            # Tratamento da predição de classe (0: Real, 1: Fake)
            if hasattr(self.model, "predict"):
                pred_class = int(self.model.predict(X)[0])
            else:
                pred_class = 1 if decision > 0 else 0
            
            # Aproximação sigmoide legada, sem calibração aprendida:
            # decision > 0 => Fake News (classe 1)
            # decision < 0 => Notícia Real (classe 0)
            # Probabilidade de Confiável = 1 / (1 + exp(decision))
            clamped_margin = float(np.clip(decision, -15.0, 15.0))
            prob_confiavel = 1.0 / (1.0 + np.exp(clamped_margin))
            ml_score = round(prob_confiavel * 100.0, 1)

            pred_label = "Notícia Provavelmente Verdadeira" if pred_class == 0 else "Possível Fake News Detectada"

            ml_details = {
                "model_type": "LinearSVC (FakeRecogna)",
                "probability_method": "uncalibrated_sigmoid",
                "hyperplane_distance": round(decision, 4),
                "predicted_class": pred_class,
                "raw_prediction_label": pred_label,
                "ml_confidence_percent": ml_score
            }
            return ml_score, ml_details

        # 3. Caso o modelo tenha apenas predict
        elif hasattr(self.model, "predict"):
            pred = self.model.predict(X)[0]
            if isinstance(pred, (int, np.integer, float, np.floating)):
                ml_score = 85.0 if pred == 0 else 15.0
                pred_class = int(pred)
            elif isinstance(pred, str):
                pred_class = 0 if pred.lower() in ("real", "true", "confiavel") else 1
                ml_score = 85.0 if pred_class == 0 else 15.0
            else:
                ml_score = 50.0
                pred_class = 0

            ml_details = {
                "model_type": "Discrete Classifier",
                "predicted_class": pred_class,
                "raw_prediction_label": "Real" if pred_class == 0 else "Fake",
                "ml_confidence_percent": ml_score
            }
            return ml_score, ml_details

        return 50.0, {"model_type": "Baseline", "ml_confidence_percent": 50.0}

    # ----------------------------------------------------
    # MÉTODOS DE AVALIAÇÃO DOS 5 CRITÉRIOS
    # ----------------------------------------------------

    def _evaluate_evidencias(self, text: str) -> dict:
        """
        Critério 1: EVIDÊNCIAS
        Avalia a presença de dados numéricos, percentuais, estatísticas,
        estudos/pesquisas citados e aspas de declarações verificáveis.
        """
        score = 50.0
        details = []

        # Detecção de citações entre aspas
        quotes = re.findall(r'["“«][^"”»]{10,}["”»]', text)
        if len(quotes) >= 3:
            score += 20.0
            details.append(f"Apresenta {len(quotes)} citações diretas de declarações.")
        elif len(quotes) >= 1:
            score += 10.0
            details.append("Apresenta declaração direta entre aspas.")
        else:
            score -= 15.0
            details.append("Poucas ou nenhuma declaração direta citada.")

        # Detecção de dados estatísticos ou números percentuais
        percentages = re.findall(r'\d+(?:[.,]\d+)?\s*%', text)
        numbers = re.findall(r'\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b', text)
        
        if percentages:
            score += 15.0
            details.append(f"Cita dados percentuais ({len(percentages)} menções).")
        elif len(numbers) > 5:
            score += 10.0
            details.append("Contém dados quantitativos e numéricos.")
        else:
            details.append("Baixa densidade de dados estatísticos/numéricos.")

        # Menção a estudos, pesquisas, relatórios ou órgãos técnicos
        keywords_evidencia = [
            "pesquisa", "estudo", "segundo dados", "relatório", "índice",
            "levantamento", "universidade", "instituto", "artigo científico"
        ]
        evid_found = [kw for kw in keywords_evidencia if kw in text.lower()]
        if evid_found:
            score += 15.0
            details.append(f"Faz referência a fontes comprobatórias ({', '.join(evid_found[:3])}).")

        score = max(5.0, min(95.0, score))
        label = "Forte" if score >= 70 else ("Moderada" if score >= 50 else "Fraca / Insuficiente")

        return {
            "score": round(score, 1),
            "label": label,
            "details": " ".join(details)
        }

    def _evaluate_qualidade_fontes(self, domain: str, author: str, links_count: int, text: str) -> dict:
        """
        Critério 2: QUALIDADE DAS FONTES
        Avalia a reputação do domínio, transparência da autoria e links para fontes primárias.
        """
        score = 50.0
        details = []

        # Análise do domínio
        known_reputable_tlds = [".gov.br", ".edu.br", ".org.br", ".jus.br", ".gov", ".edu"]
        if any(domain.endswith(tld) for tld in known_reputable_tlds):
            score += 25.0
            details.append(f"Domínio oficial ou acadêmico verificado ({domain}).")
        elif any(ext in domain for ext in ["globo.com", "uol.com.br", "estadao.com.br", "folha.uol.com.br", "cnnbrasil.com.br", "bbc.com", "reuters.com", "g1.globo.com"]):
            score += 20.0
            details.append(f"Veículo de imprensa tradicional reconhecido ({domain}).")
        elif any(sus in domain for sus in ["wordpress.com", "blogspot.com", "wixsite.com", ".xyz", ".top"]):
            score -= 20.0
            details.append(f"Domínio genérico ou plataforma de blog sem credenciais editoriais ({domain}).")
        else:
            details.append(f"Domínio informativo: {domain}.")

        # Autoria
        if author and author.lower() not in ("não informado / redação", "redação", "admin", "admin-user"):
            score += 15.0
            details.append(f"Artigo assinado por autor identificado: '{author}'.")
        else:
            score -= 10.0
            details.append("Autoria individual não especificada ou genérica.")

        # Links externos para fontes primárias
        if links_count >= 3:
            score += 10.0
            details.append(f"Contém {links_count} links de referência e verificação externa.")
        elif links_count == 0:
            score -= 5.0
            details.append("Nenhum hiperlink para fontes externas.")

        score = max(5.0, min(95.0, score))
        label = "Alta" if score >= 70 else ("Média" if score >= 50 else "Baixa / Não Verificada")

        return {
            "score": round(score, 1),
            "label": label,
            "details": " ".join(details)
        }

    def _evaluate_corroboracao(self, title: str, text: str) -> dict:
        """
        Critério 3: CORROBORAÇÃO E LINGUAGEM FACTUAL
        Detecta linguagem sensacionalista, clickbait, pedidos desesperados de compartilhamento
        e acusações conspiratórias típicas de fake news desmentidas por agências de checagem.
        """
        score = 65.0
        details = []

        # Gatilhos típicos de desinformação / fake news
        sensationalist_triggers = [
            "urgente", "bomba", "chocante", "compartilhe antes que apaguem",
            "a mídia esconde", "ninguém quer que você veja", "o que ninguém te conta",
            "segredo revelado", "milagre", "cura definitiva", "absurdo", "inacreditável",
            "veja antes que saia do ar", "escândalo mundial", "conspiração"
        ]

        full_content_lower = f"{title} {text}".lower()
        found_triggers = [trig for trig in sensationalist_triggers if trig in full_content_lower]

        if found_triggers:
            penalty = min(40.0, len(found_triggers) * 15.0)
            score -= penalty
            details.append(f"Detectados termos sensacionalistas/alarmistas ({', '.join(found_triggers)}).")
        else:
            score += 15.0
            details.append("Tom da matéria sóbrio e jornalístico, sem gatilhos alarmistas.")

        # Excesso de exclamações ou letras maiúsculas no título (Clickbait)
        if "!" in title or "?" in title:
            score -= 10.0
            details.append("Título expressivo com pontuação interrogativa ou exclamativa.")
        
        words_in_caps = [w for w in title.split() if w.isupper() and len(w) > 3]
        if words_in_caps:
            score -= 10.0
            details.append(f"Palavras em caixa alta no título ({', '.join(words_in_caps)}).")

        score = max(5.0, min(95.0, score))
        label = "Consistente" if score >= 70 else ("Moderada" if score >= 50 else "Inconsistente / Sensacionalista")

        return {
            "score": round(score, 1),
            "label": label,
            "details": " ".join(details)
        }

    def _evaluate_contexto(self, title: str, text: str, word_count: int) -> dict:
        """
        Critério 4: CONTEXTO
        Avalia se o corpo do texto de fato desenvolve e condiz com o título,
        e se o artigo tem extensão suficiente para fornecer contexto histórico e fático.
        """
        score = 55.0
        details = []

        # Extensão do artigo
        if word_count >= 300:
            score += 15.0
            details.append(f"Extensão do texto aprofundada ({word_count} palavras), fornecendo contexto abrangente.")
        elif word_count >= 120:
            score += 5.0
            details.append(f"Extensão moderada ({word_count} palavras).")
        else:
            score -= 20.0
            details.append(f"Texto muito curto ({word_count} palavras), com alto risco de descontextualização.")

        # Correspondência entre título e corpo (sobreposição de palavras-chave)
        title_words = set(re.findall(r'\b[a-zA-ZáéíóúãõâêîôûçÁÉÍÓÚÃÕÂÊÎÔÛÇ]{4,}\b', title.lower()))
        text_words = set(re.findall(r'\b[a-zA-ZáéíóúãõâêîôûçÁÉÍÓÚÃÕÂÊÎÔÛÇ]{4,}\b', text.lower()))

        if title_words:
            overlap = title_words.intersection(text_words)
            overlap_ratio = len(overlap) / len(title_words)
            if overlap_ratio >= 0.7:
                score += 15.0
                details.append("Forte coerência temática entre a manchete e o desenvolvimento do texto.")
            elif overlap_ratio < 0.4:
                score -= 15.0
                details.append("Possível descompasso entre a chamada (título) e o conteúdo efetivo.")

        score = max(5.0, min(95.0, score))
        label = "Adequado" if score >= 70 else ("Parcial" if score >= 50 else "Frágil / Descontextualizado")

        return {
            "score": round(score, 1),
            "label": label,
            "details": " ".join(details)
        }

    def _evaluate_atualidade(self, publish_date: str, text: str) -> dict:
        """
        Critério 5: ATUALIDADE
        Avalia se a notícia possui datação clara e se o evento é contemporâneo
        (evita recirculação de notícias antigas fora de contexto).
        """
        score = 50.0
        details = []

        if publish_date:
            try:
                dt = date_parser.parse(publish_date, fuzzy=True)
                now = datetime.now()
                # Se o ano for recente (últimos 2 anos)
                days_diff = (now - dt.replace(tzinfo=None)).days
                
                if days_diff <= 30:
                    score += 35.0
                    details.append(f"Matéria recente (publicada há cerca de {days_diff} dias).")
                elif days_diff <= 365:
                    score += 20.0
                    details.append(f"Matéria publicada no último ano ({dt.strftime('%d/%m/%Y')}).")
                else:
                    score -= 10.0
                    details.append(f"Artigo com mais de 1 ano de publicação ({dt.strftime('%d/%m/%Y')}); verificar se não se trata de reaproveitamento de fatos antigos.")
            except Exception:
                score += 10.0
                details.append(f"Data identificada no cabeçalho: '{publish_date}'.")
        else:
            # Procura menção a anos no texto
            current_year = datetime.now().year
            recent_years = [str(current_year), str(current_year - 1), str(current_year - 2)]
            if any(y in text for y in recent_years):
                score += 15.0
                details.append(f"Menção a fatos do período recente ({', '.join(recent_years)}), embora sem timestamp exato.")
            else:
                score -= 15.0
                details.append("Sem data de publicação explícita ou menção a períodos contemporâneos.")

        score = max(5.0, min(95.0, score))
        label = "Recente / Verificado" if score >= 70 else ("Moderada" if score >= 50 else "Indeterminada / Desatualizada")

        return {
            "score": round(score, 1),
            "label": label,
            "details": " ".join(details)
        }

    def _generate_summary(self, score: float, verdict: str, evid: dict, fontes: dict, corrob: dict) -> str:
        """Gera uma síntese concisa do resultado da análise."""
        if verdict == "confiavel":
            return (
                f"A notícia apresenta índice de confiabilidade favorável de {score}%. "
                f"Apresenta qualidade de fontes avaliada como '{fontes['label']}' e nível de evidências '{evid['label']}', "
                f"com discurso alinhado aos padrões informativos."
            )
        else:
            return (
                f"Atenção: A notícia obteve índice de confiabilidade baixo de {score}%, classificada como suspeita ou desconfiável. "
                f"O nível de evidências foi avaliado como '{evid['label']}' e a corroboração como '{corrob['label']}'. "
                f"Recomenda-se checagem em agências independentes antes de compartilhar."
            )

