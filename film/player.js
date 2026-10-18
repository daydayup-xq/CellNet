import { createFilm, DURATION, SCENES } from './film-renderer.js';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search), exporting = params.get('export') === '1';
if (exporting) document.documentElement.classList.add('export');
let language = params.get('lang') === 'zh' ? 'zh' : 'en';
if (!params.has('lang')) { try { language = localStorage.getItem('cellnet-language') === 'zh' ? 'zh' : 'en'; } catch {} }
let renderer, time=0, playing=false, frame=0, sceneIndex=-1, started=false, playRequest=0, audioLanguage='';
const canvas=$('film-canvas'), soundtrack=$('soundtrack'), reduced=matchMedia('(prefers-reduced-motion: reduce)');
const t=(en,zh)=>language==='zh'?zh:en;
const format=value=>`${String(Math.floor(value/60)).padStart(2,'0')}:${String(Math.floor(value%60)).padStart(2,'0')}`;
const sceneAt=value=>Math.max(0,SCENES.findIndex(s=>value>=s.start&&(value<s.end||s.end===DURATION)));
const chapterNames={
  en:['Panoramic modelling','DNA & expression','Shared learning','Cellular diversity','Expression prediction','Molecular programmes','CellNet'],
  zh:['全景建模','DNA 与表达','共享学习','细胞多样性','表达预测','分子程序','CellNet']
};
const narration={
  en:['Cellular complexity calls for a panoramic view.','CellNet combines DNA identity with RNA expression. Coding and non-coding genes, in one image.','Across species and masked views, cellular context meets molecular detail.','Recognize rare populations. Resolve cell states.','Predict gene expression from observed context. Across species and spatial measurements.','Query RNA. Connect molecular modules and biological programmes.','CellNet. Panoramic transcriptome modelling.'],
  zh:['细胞的复杂性，需要全景的视角。','CellNet，融合 DNA 身份与 RNA 表达。让编码与非编码基因，进入同一幅图像。','跨物种学习，保留完整视野。连接细胞全景与分子细节。','识别稀有群体，分辨细胞状态。','从已观测表达，预测基因表达，拓展至跨物种与空间测量。','以 RNA 为查询，连接分子模块与生物学程序。','CellNet，转录组的全景建模。']
};
function updateControls(){
  $('time').value=`${format(time)} / ${format(DURATION)}`;
  $('seek').value=String(time);
  $('seek').setAttribute('aria-valuetext',t(`${Math.round(time)} seconds of ${DURATION} seconds`,`${Math.round(time)} 秒，共 ${DURATION} 秒`));
  $('play-icon').textContent=playing?'Ⅱ':'▶';
  $('play').setAttribute('aria-label',playing?t('Pause','暂停'):time>=DURATION?t('Replay','重播'):t('Play with sound','有声播放'));
  $('start').hidden=started||playing||!renderer||exporting;
  $('sound').setAttribute('aria-label',soundtrack.muted?t('Unmute','开启声音'):t('Mute','静音'));
  $('sound').setAttribute('aria-pressed',String(soundtrack.muted));
  $('sound-waves').style.display=soundtrack.muted?'none':'';
  $('sound-off').style.display=soundtrack.muted?'':'none';
  const next=sceneAt(time);
  if(next!==sceneIndex){sceneIndex=next;document.querySelectorAll('.chapter').forEach((el,i)=>el.setAttribute('aria-current',String(i===next)));$('chapter-label').textContent=chapterNames[language][next];$('scene-announcement').textContent=SCENES[next].title[language];}
  canvas.setAttribute('aria-label',`${SCENES[next].title[language]}. ${SCENES[next].subtitle[language]}`);
}
function draw(){renderer?.render(time,language);updateControls();}
function pause(){
  if(playing&&Number.isFinite(soundtrack.currentTime))time=Math.min(DURATION,soundtrack.currentTime);
  playing=false;playRequest++;soundtrack.pause();cancelAnimationFrame(frame);draw();
}
function tick(){
  if(!playing)return;
  // The soundtrack clock keeps narration, music and motion together during buffering.
  time=Math.min(DURATION,soundtrack.currentTime);draw();
  if(time>=DURATION||soundtrack.ended){pause();time=DURATION;draw();return;}
  frame=requestAnimationFrame(tick);
}
async function play(){
  if(!renderer||playing||exporting)return;
  if(time>=DURATION)seek(0);
  const request=++playRequest;
  soundtrack.currentTime=time;playing=true;started=true;$('audio-status').hidden=true;updateControls();
  try{await soundtrack.play();if(request!==playRequest||!playing)return;frame=requestAnimationFrame(tick);}
  catch(error){
    if(request!==playRequest)return;
    playing=false;started=false;
    $('audio-status').textContent=error.name==='NotAllowedError'?t('Press play to begin with sound.','点击播放，开启声音。'):t('Audio could not load. Retry, or download the MP4 below.','声音加载未完成，请重试或下载下方 MP4。');
    $('audio-status').hidden=false;updateControls();
  }
}
function seek(value){
  const number=Number(value);if(!Number.isFinite(number))return;
  time=Math.min(DURATION,Math.max(0,number));if(time>0)started=true;
  if(!exporting&&audioLanguage)soundtrack.currentTime=time;
  draw();
}
function setLanguage(value){
  const next=value==='zh'?'zh':'en',resume=playing&&next!==language;if(resume)pause();language=next;
  document.documentElement.lang=language==='zh'?'zh-CN':'en';
  document.querySelectorAll('[data-en][data-zh]').forEach(el=>{el.textContent=el.dataset[language];});
  $('language').textContent=language==='zh'?'EN':'中文';$('language').setAttribute('aria-label',t('Switch to Chinese','Switch to English'));
  $('replay').setAttribute('aria-label',t('Replay from the beginning','从头重播'));
  $('start').setAttribute('aria-label',t('Play with narration and music','播放配音与音乐'));
  $('screen').setAttribute('aria-label',t('Film canvas. Space to play or pause; arrow keys to seek.','短片画面。空格播放或暂停，方向键调整进度。'));
  $('fullscreen').setAttribute('aria-label',document.fullscreenElement?t('Exit fullscreen','退出全屏'):t('Enter fullscreen','进入全屏'));
  $('chapters').setAttribute('aria-label',t('Film chapters','短片章节'));
  $('video-download').href=`cellnet-film-${language}.mp4`;
  document.title=t('CellNet — Panoramic transcriptome modelling · The film','CellNet — 转录组的全景建模 · 研究短片');
  if(!exporting&&audioLanguage!==language){soundtrack.src=`audio/soundtrack-${language}.m4a`;audioLanguage=language;soundtrack.load();soundtrack.currentTime=time;}
  const chapters=document.createDocumentFragment(),transcript=document.createDocumentFragment();
  SCENES.forEach((scene,index)=>{
    const button=document.createElement('button');button.type='button';button.className='chapter';
    const stamp=document.createElement('span');stamp.textContent=format(scene.start);
    const name=document.createElement('strong');name.textContent=chapterNames[language][index];button.append(stamp,name);
    button.addEventListener('click',()=>{pause();seek(scene.start);});chapters.append(button);
    const item=document.createElement('li'),jump=document.createElement('button'),copy=document.createElement('div'),heading=document.createElement('h2'),paragraph=document.createElement('p');
    jump.type='button';jump.textContent=format(scene.start);jump.setAttribute('aria-label',t(`Seek to ${scene.title.en}`,`跳至${scene.title.zh}`));
    jump.addEventListener('click',()=>{pause();seek(scene.start);$('screen').focus();});
    heading.textContent=scene.title[language];paragraph.textContent=narration[language][index];
    if(scene.id==='prediction')paragraph.textContent+=t(' Evaluations cover coding and selected non-coding targets, with target expression withheld from the input.',' 评估覆盖编码与部分非编码靶标，靶标表达从输入中留出。');
    copy.append(heading,paragraph);item.append(jump,copy);transcript.append(item);
  });
  $('chapters').replaceChildren(chapters);$('transcript-list').replaceChildren(transcript);sceneIndex=-1;draw();if(resume)return play();
}
$('seek').max=String(DURATION);
$('play').addEventListener('click',()=>playing?pause():play());
$('start').addEventListener('click',event=>{event.stopPropagation();play();});
$('screen').addEventListener('click',()=>playing?pause():play());
$('replay').addEventListener('click',()=>{pause();seek(0);play();});
$('seek').addEventListener('input',event=>seek(event.target.value));
$('sound').addEventListener('click',()=>{soundtrack.muted=!soundtrack.muted;updateControls();});
$('language').addEventListener('click',()=>{setLanguage(language==='en'?'zh':'en');try{localStorage.setItem('cellnet-language',language);}catch{}const url=new URL(location.href);url.searchParams.set('lang',language);history.replaceState({},'',url);});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('film').requestFullscreen();}catch{}});
if(!document.fullscreenEnabled)$('fullscreen').hidden=true;
document.addEventListener('fullscreenchange',()=>$('fullscreen').setAttribute('aria-label',document.fullscreenElement?t('Exit fullscreen','退出全屏'):t('Enter fullscreen','进入全屏')));
$('film').addEventListener('keydown',event=>{
  if(event.target.matches('input,button,a'))return;
  if(event.code==='Space'||event.key.toLowerCase()==='k'){event.preventDefault();playing?pause():play();}
  if(event.key.toLowerCase()==='m'){event.preventDefault();soundtrack.muted=!soundtrack.muted;updateControls();}
  if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();seek(time+(event.key==='ArrowRight'?5:-5));}
  if(event.key==='Home'){event.preventDefault();seek(0);}
  if(event.key==='End'){event.preventDefault();pause();seek(DURATION);}
});
soundtrack.addEventListener('ended',()=>{pause();time=DURATION;draw();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
const ready=(async()=>{
  await Promise.all([document.fonts.load('400 36px "DM Sans"'),document.fonts.load('400 78px "Instrument Serif"'),document.fonts.load('italic 78px "Instrument Serif"'),document.fonts.load('500 36px "Noto Sans SC"','细胞全景基因表达')]);
  await document.fonts.ready;renderer=await createFilm(canvas);$('loading').hidden=true;
  ['play','replay','seek','sound'].forEach(id=>{$(id).disabled=false;});setLanguage(language);
  if(params.get('autoplay')==='1'&&!reduced.matches&&!exporting)await play();return true;
})();
ready.catch(error=>{$('loading').textContent=t('The film could not load. Please reload, or download the MP4 below.','短片加载未完成，请刷新页面或下载下方 MP4。');console.error(error);});
window.cellnetFilm={ready,duration:DURATION,canvas,soundtrack,play,pause,seek,setLanguage,render:(value,lang=language)=>{pause();if(lang!==language)setLanguage(lang);seek(value);},get time(){return time;},get playing(){return playing;},get language(){return language;}};
