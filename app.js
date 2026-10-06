const CRIT=[
 {n:"Evidências",w:39.8,d:"Dados, provas e referências consultáveis."},
 {n:"Qualidade das Fontes",w:24.1,d:"Credibilidade e reputação das fontes."},
 {n:"Corroboração",w:17.1,d:"Confirmação por fontes independentes."},
 {n:"Contexto",w:11.8,d:"Sem recortes que alterem o sentido."},
 {n:"Atualidade",w:7.2,d:"Recência e relevância temporal."}];
// Protótipo: pontuações simuladas. Troque por chamada à sua API.
function scoresFor(url){
  if(url.includes("kaggle.com"))return[24,19,5,33,40];
  let h=0;for(const c of url)h=(h*31+c.charCodeAt(0))>>>0;
  return CRIT.map((_,i)=>Math.round(8+((h>>>(i*5))%85)));
}
const color=v=>v<35?"var(--red)":v<65?"var(--amber)":"var(--green)";
const $=id=>document.getElementById(id);
const fmt=n=>String(n).replace(".",",");
function show(url){
  let host;try{host=new URL(url).hostname.replace(/^www\./,"")}catch{return false}
  const s=scoresFor(url);
  const total=Math.round(CRIT.reduce((a,c,i)=>a+c.w/100*s[i],0));
  $("rows").innerHTML=CRIT.map((c,i)=>`<div class="row"><div class="top"><span><b>${c.n}</b> <i>· peso ${fmt(c.w)}%</i></span><b>${s[i]}</b></div><div class="bar"><div data-w="${s[i]}" style="background:${color(s[i])}"></div></div><div class="desc">${c.d}</div></div>`).join("");
  const hi=s.indexOf(Math.max(...s)),lo=s.indexOf(Math.min(...s));
  $("best").innerHTML=`Ponto mais forte: <b>${CRIT[hi].n}</b> (${s[hi]}). ${CRIT[hi].d}`;
  $("worst").innerHTML=`Ponto mais fraco: <b>${CRIT[lo].n}</b> (${s[lo]}). Vale investigar este aspecto.`;
  $("unc").textContent=s[2]<40?"Poucas fontes independentes confirmaram o fato até agora.":"Verifique por conta própria os dados citados na matéria.";
  const v=$("verdict");v.textContent=total<40?"Baixa confiabilidade estimada":total<70?"Confiabilidade moderada estimada":"Alta confiabilidade estimada";v.style.color=color(total);
  $("meta").textContent=`${host} · analisado em ${new Date().toLocaleDateString("pt-BR")}`;
  $("home").style.display="none";$("result").style.display="block";window.scrollTo(0,0);
  $("score").textContent=total;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    document.querySelectorAll(".bar div").forEach(b=>b.style.width=b.dataset.w+"%");
    $("arc").setAttribute("stroke-dasharray",`${total} 100`);
  }));
  return true;
}
$("form").addEventListener("submit",e=>{
  e.preventDefault();
  let u=$("url").value.trim();
  if(u&&!/^https?:\/\//i.test(u))u="https://"+u;
  if(!u||!show(u))$("err").textContent="Cole um link completo, como https://exemplo.com.br/noticia";
  else $("err").textContent="";
});
$("back").onclick=()=>{$("result").style.display="none";$("home").style.display="block";$("url").value="";$("url").focus()};
