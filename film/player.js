import { createFilm, DURATION, SCENES } from './film-renderer.js?v=4';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search), exporting = params.get('export') === '1';
if (exporting) document.documentElement.classList.add('export');
// One English visual edition, regardless of the homepage locale or legacy URLs.
document.documentElement.lang='en';
let renderer, time=0, playing=false, frame=0, sceneIndex=-1, started=false, playRequest=0;
const canvas=$('film-canvas'), soundtrack=$('soundtrack'), reduced=matchMedia('(prefers-reduced-motion: reduce)');
const format=value=>String(Math.floor(value/60)).padStart(2,'0')+':'+String(Math.floor(value%60)).padStart(2,'0');
const sceneAt=value=>Math.max(0,SCENES.findIndex(s=>value>=s.start&&(value<s.end||s.end===DURATION)));
const english=value=>typeof value==='string'?value:value.en;
const chapterNames=['Panoramic modelling','A wider gene vocabulary','Image Atlas','Practical panoramic learning','Cell identities & states','A wider gene space','Non-coding programmes','Two foundations'];
function updateControls(){
  $('time').value=format(time)+' / '+format(DURATION);
  $('seek').value=String(time);
  $('seek').setAttribute('aria-valuetext',Math.round(time)+' seconds of '+DURATION+' seconds');
  $('play-icon').textContent=playing?'Ⅱ':'▶';
  $('play').setAttribute('aria-label',playing?'Pause':time>=DURATION?'Replay':'Play with music');
  $('start').hidden=started||playing||!renderer||exporting;
  $('sound').setAttribute('aria-label',soundtrack.muted?'Unmute':'Mute');
  $('sound').setAttribute('aria-pressed',String(soundtrack.muted));
  $('sound-waves').style.display=soundtrack.muted?'none':'';
  $('sound-off').style.display=soundtrack.muted?'':'none';
  const next=sceneAt(time);
  if(next!==sceneIndex){sceneIndex=next;document.querySelectorAll('.chapter').forEach((el,i)=>el.setAttribute('aria-current',String(i===next)));$('chapter-label').textContent=chapterNames[next];$('scene-announcement').textContent=english(SCENES[next].title);}
  canvas.setAttribute('aria-label',english(SCENES[next].title)+'. '+english(SCENES[next].subtitle));
}
function draw(){renderer?.render(time,'en');updateControls();}
function pause(){
  if(playing&&Number.isFinite(soundtrack.currentTime))time=Math.min(DURATION,soundtrack.currentTime);
  playing=false;playRequest++;soundtrack.pause();cancelAnimationFrame(frame);draw();
}
function tick(){
  if(!playing)return;
  // The music clock keeps every motion cue on the beat, including after buffering.
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
    $('audio-status').textContent=error.name==='NotAllowedError'?'Press play to begin with music.':'Music could not load. Retry, or download the MP4 below.';
    $('audio-status').hidden=false;updateControls();
  }
}
function seek(value){
  const number=Number(value);if(!Number.isFinite(number))return;
  time=Math.min(DURATION,Math.max(0,number));if(time>0)started=true;
  if(!exporting)soundtrack.currentTime=time;
  draw();
}
// Compatibility for deterministic exporters; the film is always English.
function setLanguage(){draw();}
function chapters(){
  const buttons=document.createDocumentFragment(),story=document.createDocumentFragment();
  SCENES.forEach((scene,index)=>{
    const button=document.createElement('button');button.type='button';button.className='chapter';
    const stamp=document.createElement('span');stamp.textContent=format(scene.start);
    const name=document.createElement('strong');name.textContent=chapterNames[index];button.append(stamp,name);
    button.addEventListener('click',()=>{pause();seek(scene.start);});buttons.append(button);
    const item=document.createElement('li'),jump=document.createElement('button'),copy=document.createElement('div'),heading=document.createElement('h2'),paragraph=document.createElement('p');
    jump.type='button';jump.textContent=format(scene.start);jump.setAttribute('aria-label','Seek to '+english(scene.title));
    jump.addEventListener('click',()=>{pause();seek(scene.start);$('screen').focus();});
    heading.textContent=english(scene.title);paragraph.textContent=english(scene.subtitle);
    copy.append(heading,paragraph);item.append(jump,copy);story.append(item);
  });
  $('chapters').replaceChildren(buttons);$('transcript-list').replaceChildren(story);
}
$('seek').max=String(DURATION);
$('play').addEventListener('click',()=>playing?pause():play());
$('start').addEventListener('click',event=>{event.stopPropagation();play();});
$('screen').addEventListener('click',()=>playing?pause():play());
$('replay').addEventListener('click',()=>{pause();seek(0);play();});
$('seek').addEventListener('input',event=>seek(event.target.value));
$('sound').addEventListener('click',()=>{soundtrack.muted=!soundtrack.muted;updateControls();});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('film').requestFullscreen();}catch{}});
if(!document.fullscreenEnabled)$('fullscreen').hidden=true;
document.addEventListener('fullscreenchange',()=>$('fullscreen').setAttribute('aria-label',document.fullscreenElement?'Exit fullscreen':'Enter fullscreen'));
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
  await Promise.all([document.fonts.load('400 36px "DM Sans"'),document.fonts.load('700 100px "DM Sans"'),document.fonts.load('400 78px "Instrument Serif"'),document.fonts.load('italic 78px "Instrument Serif"')]);
  await document.fonts.ready;renderer=await createFilm(canvas);$('loading').hidden=true;
  if(!exporting){soundtrack.src='audio/score-v3.m4a';soundtrack.load();}
  chapters();['play','replay','seek','sound'].forEach(id=>{$(id).disabled=false;});draw();
  if(params.get('autoplay')==='1'&&!reduced.matches&&!exporting)await play();return true;
})();
ready.catch(error=>{$('loading').textContent='The film could not load. Please reload, or download the MP4 below.';console.error(error);});
window.cellnetFilm={ready,duration:DURATION,canvas,soundtrack,play,pause,seek,setLanguage,render:value=>{pause();seek(value);},get time(){return time;},get playing(){return playing;},get language(){return 'en';}};
