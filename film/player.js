import { createFilm, DURATION, SCENES } from './film-renderer.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const exporting = params.get('export') === '1';
if (exporting) document.documentElement.classList.add('export');
let language = params.get('lang') === 'zh' ? 'zh' : 'en';
if (!params.has('lang')) {
  try { language = localStorage.getItem('cellnet-language') === 'zh' ? 'zh' : 'en'; } catch {}
}
let renderer, time = 0, playing = false, lastTick = 0, frame = 0, sceneIndex = -1, started = false;
const canvas = $('film-canvas');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const t = (en, zh) => language === 'zh' ? zh : en;
const format = value => `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
const sceneAt = value => Math.max(0, SCENES.findIndex(scene => value >= scene.start && (value < scene.end || scene.end === DURATION)));
const chapterNames = {
  en:['Panoramic modelling','DNA & expression','Panoramic learning','Across species','Cellular diversity','Expression prediction','Molecular programmes','CellNet'],
  zh:['全景建模','DNA 与表达','全景学习','跨物种学习','细胞多样性','表达预测','分子程序','CellNet']
};
const storyDetails = {
  en:[
    'Single-cell assays reveal an increasingly rich transcriptome, including RNAs without a protein product.',
    'Each gene occupies a pixel in a 384 × 384 canvas. Gene positions stay fixed across cells of the same species. A measurement mask distinguishes measured zeros from unmeasured genes.',
    'Two lightly masked and six strongly masked views retain the full image extent. The shared encoder learns cellular context; a target gene’s DNA identity queries that context to predict detection and expression magnitude.',
    'A shared model learns from transcriptome images across species. The displayed images are real examples from mouse, zebrafish and fruit fly.',
    'Evaluations include donor-held-out cellular identification, rare kidney and lung populations, T-cell polarization and fibroblast senescence.',
    'The target expression is withheld from the input context. The study evaluates coding and selected non-coding targets, cross-species prediction and recovery of spatial expression patterns, including targets without expression-pretraining exposure whose DNA identities remain available.',
    'Reported retrievals include developmental and neural programmes. The sparsely detected non-coding RNA SERTAD4-AS1 retrieves WNT5B, DLK1 and SERPINF1 among its nearest neighbours in the study.',
    'Explore the method, real transcriptome images, model weights and open-source code through the project website.'
  ],
  zh:[
    '单细胞测量正揭示越来越丰富的转录组，其中包含不产生蛋白质的 RNA。',
    '每个基因在 384 × 384 画布中占据一个像素，同一物种的细胞共享固定基因位置。测量掩码区分实测零值与未测量基因。',
    '两个轻掩码视图与六个强掩码视图保留完整图像范围。共享编码器学习细胞语境；靶基因的 DNA 身份查询该语境，预测检出概率与表达强度。',
    '同一模型从跨物种转录组图像中学习；画面中的图像来自小鼠、斑马鱼与果蝇的真实样例。',
    '评估覆盖供者留出的细胞识别、肾脏与肺脏稀有群体、T 细胞极化及成纤维细胞衰老。',
    '靶标表达从输入语境中留出。研究评估了编码及部分非编码靶标、跨物种预测与空间表达模式恢复，也包括未参与表达预训练、但仍具有 DNA 身份的靶标。',
    '研究检索了发育与神经相关程序。例如，低检出率的非编码 RNA SERTAD4-AS1 在近邻中检索到 WNT5B、DLK1 和 SERPINF1。',
    '通过项目网站探索方法、真实转录组图像、模型权重与开源代码。'
  ]
};

function updateControls() {
  $('time').value = `${format(time)} / ${format(DURATION)}`;
  $('seek').value = String(time);
  $('seek').setAttribute('aria-valuetext', t(`${Math.round(time)} seconds of 60 seconds`, `${Math.round(time)} 秒，共 60 秒`));
  $('play-icon').textContent = playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label', playing ? t('Pause', '暂停') : time >= DURATION ? t('Replay', '重播') : t('Play', '播放'));
  $('start').hidden = started || playing || !renderer || exporting;
  const next = sceneAt(time);
  if (next !== sceneIndex) {
    sceneIndex = next;
    document.querySelectorAll('.chapter').forEach((el, i) => el.setAttribute('aria-current', String(i === next)));
    $('chapter-label').textContent = chapterNames[language][next];
    $('scene-announcement').textContent = SCENES[next].title[language];
  }
  canvas.setAttribute('aria-label', `${SCENES[next].title[language]}. ${SCENES[next].subtitle[language]}`);
}
function draw() { renderer?.render(time, language); updateControls(); }
function pause() { playing = false; cancelAnimationFrame(frame); updateControls(); }
function tick(now) {
  if (!playing) return;
  time = Math.min(DURATION, time + Math.max(0, now - lastTick) / 1000);
  lastTick = now;
  draw();
  if (time >= DURATION) { pause(); return; }
  frame = requestAnimationFrame(tick);
}
function play() {
  if (!renderer || playing) return;
  if (time >= DURATION) time = 0;
  started = true; playing = true; lastTick = performance.now();
  updateControls(); frame = requestAnimationFrame(tick);
}
function seek(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return;
  time = Math.min(DURATION, Math.max(0, number));
  if (time > 0) started = true;
  lastTick = performance.now(); draw();
}
function setLanguage(value) {
  language = value === 'zh' ? 'zh' : 'en';
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-en][data-zh]').forEach(el => { el.textContent = el.dataset[language]; });
  $('language').textContent = language === 'zh' ? 'EN' : '中文';
  $('language').setAttribute('aria-label', t('Switch to Chinese', 'Switch to English'));
  $('replay').setAttribute('aria-label', t('Replay from the beginning', '从头重播'));
  $('start').setAttribute('aria-label', t('Play the film', '播放短片'));
  $('screen').setAttribute('aria-label', t('Film canvas. Space to play or pause; arrow keys to seek.', '短片画面。空格播放或暂停，方向键调整进度。'));
  $('fullscreen').setAttribute('aria-label', document.fullscreenElement ? t('Exit fullscreen', '退出全屏') : t('Enter fullscreen', '进入全屏'));
  $('chapters').setAttribute('aria-label', t('Film chapters', '短片章节'));
  $('video-download').href = `cellnet-film-${language}.mp4`;
  document.title = t('CellNet — A panoramic view of the cell · The film', 'CellNet — 细胞的全景视角 · 研究短片');
  const chapters = document.createDocumentFragment(), transcript = document.createDocumentFragment();
  SCENES.forEach((scene, index) => {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'chapter';
    const stamp = document.createElement('span'); stamp.textContent = format(scene.start);
    const name = document.createElement('strong'); name.textContent = chapterNames[language][index];
    button.append(stamp, name);
    button.addEventListener('click', () => { pause(); seek(scene.start + 0.8); });
    chapters.append(button);
    const item = document.createElement('li'), jump = document.createElement('button'), copy = document.createElement('div'), heading = document.createElement('h2'), paragraph = document.createElement('p');
    jump.type = 'button'; jump.textContent = format(scene.start); jump.setAttribute('aria-label', t(`Seek to ${scene.title.en}`, `跳至${scene.title.zh}`));
    jump.addEventListener('click', () => { pause(); seek(scene.start + .8); $('screen').focus(); });
    heading.textContent = scene.title[language]; paragraph.textContent = `${scene.subtitle[language]} ${storyDetails[language][index]}`;
    copy.append(heading, paragraph); item.append(jump, copy); transcript.append(item);
  });
  $('chapters').replaceChildren(chapters); $('transcript-list').replaceChildren(transcript);
  sceneIndex = -1; draw();
}

$('play').addEventListener('click', () => playing ? pause() : play());
$('start').addEventListener('click', event => { event.stopPropagation(); play(); });
$('screen').addEventListener('click', () => playing ? pause() : play());
$('replay').addEventListener('click', () => { seek(0); play(); });
$('seek').addEventListener('input', event => seek(event.target.value));
$('language').addEventListener('click', () => {
  setLanguage(language === 'en' ? 'zh' : 'en');
  try { localStorage.setItem('cellnet-language', language); } catch {}
  const url = new URL(location.href); url.searchParams.set('lang', language); history.replaceState({}, '', url);
});
$('fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('film').requestFullscreen(); } catch { /* Browsers without fullscreen still provide the responsive player. */ }
});
if (!document.fullscreenEnabled) $('fullscreen').hidden = true;
document.addEventListener('fullscreenchange', () => $('fullscreen').setAttribute('aria-label', document.fullscreenElement ? t('Exit fullscreen', '退出全屏') : t('Enter fullscreen', '进入全屏')));
$('film').addEventListener('keydown', event => {
  if (event.target.matches('input,button,a')) return;
  if (event.code === 'Space' || event.key.toLowerCase() === 'k') { event.preventDefault(); playing ? pause() : play(); }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); seek(time + (event.key === 'ArrowRight' ? 5 : -5)); }
  if (event.key === 'Home') { event.preventDefault(); seek(0); }
  if (event.key === 'End') { event.preventDefault(); seek(DURATION); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });

const ready = (async () => {
  await Promise.all([document.fonts.load('400 36px "DM Sans"'), document.fonts.load('400 78px "Instrument Serif"'), document.fonts.load('italic 78px "Instrument Serif"'), document.fonts.load('400 36px "Noto Sans SC"', '细胞全景基因表达')]);
  await document.fonts.ready;
  renderer = await createFilm(canvas);
  $('loading').hidden = true;
  ['play','replay','seek'].forEach(id => { $(id).disabled = false; });
  setLanguage(language);
  if (params.get('autoplay') === '1' && !reduced.matches && !exporting) play();
  return true;
})();
ready.catch(error => { $('loading').textContent = t('The film could not load. Please reload, or download the MP4 below.', '短片加载未完成，请刷新页面或下载下方 MP4。'); console.error(error); });
window.cellnetFilm = { ready, duration:DURATION, canvas, play, pause, seek, setLanguage, render:(value, lang=language) => { pause(); if(lang !== language) setLanguage(lang); seek(value); }, get time(){ return time; }, get playing(){ return playing; }, get language(){ return language; } };
