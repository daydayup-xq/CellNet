(() => {
  'use strict';
  let language = 'en';
  try { language = localStorage.getItem('cellnet-language') === 'zh' ? 'zh' : 'en'; } catch {}
  let cells = [], species = 'all', expanded = false, currentCell = null;
  const grid = document.getElementById('gallery-grid');
  const more = document.getElementById('show-more');
  const mask = document.getElementById('mask-toggle');
  const dialog = document.getElementById('cell-dialog');
  const linear = document.getElementById('linear-toggle');
  const t = (en, zh) => language === 'zh' ? zh : en;
  const speciesName = value => ({ Mouse:t('Mouse','小鼠'), Zebrafish:t('Zebrafish','斑马鱼'), Fruit_fly:t('Fruit fly','果蝇') })[value] || value;
  const number = value => Number(value).toLocaleString(language === 'zh' ? 'zh-CN' : 'en-US');
  function applyLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(el => { el.textContent = el.dataset[language]; });
    document.querySelectorAll('[data-film-link]').forEach(el => { el.href = 'film/?autoplay=1&v=7'; });
    const button = document.getElementById('language-button');
    button.textContent = language === 'zh' ? 'EN' : '中文';
    button.setAttribute('aria-label',language === 'zh' ? 'Switch to English' : '切换至中文');
    document.getElementById('menu-button').setAttribute('aria-label', t('Toggle navigation','展开或收起导航'));
    document.getElementById('dialog-close').setAttribute('aria-label', t('Close image details','关闭图像详情'));
    document.title = t('CellNet: A panoramic view of the cell','CellNet: 单细胞转录组的全景视角');
    if(cells.length) renderGallery();
    if(currentCell) updateDialog();
  }
  document.getElementById('language-button').addEventListener('click', () => {
    language = language === 'en' ? 'zh' : 'en';
    try { localStorage.setItem('cellnet-language',language); } catch {}
    applyLanguage();
  });
  const menu = document.getElementById('menu-button'), mobileNav = document.getElementById('mobile-nav');
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') === 'true';
    menu.setAttribute('aria-expanded',String(!open)); mobileNav.hidden = open;
  });
  mobileNav.querySelectorAll('a').forEach(a => a.addEventListener('click', () => { mobileNav.hidden = true; menu.setAttribute('aria-expanded','false'); }));
  document.addEventListener('keydown', e => { if(e.key === 'Escape') { mobileNav.hidden = true; menu.setAttribute('aria-expanded','false'); } });
  function node(tag, className, content) {
    const element = document.createElement(tag);
    if(className) element.className = className;
    if(content !== undefined) element.textContent = content;
    return element;
  }
  function renderGallery() {
    const filtered = species === 'all' ? cells : cells.filter(cell => cell.species === species);
    const initial = species === 'all' ? cells.filter((_, index) => index % 8 < 4) : filtered;
    const visible = expanded || species !== 'all' ? filtered : initial;
    const fragment = document.createDocumentFragment();
    visible.forEach(cell => {
      const tile = node('button','cell-tile'); tile.type = 'button';
      tile.setAttribute('aria-label',t(`Inspect ${speciesName(cell.species)} ${cell.cell_type}, ${cell.id}`,`查看${speciesName(cell.species)} ${cell.cell_type}，${cell.id}`));
      const imageBox = node('div','tile-image');
      const img = node('img'); img.src = mask.checked ? cell.mask_image : cell.image;
      img.alt = `${speciesName(cell.species)} ${cell.cell_type} ${mask.checked ? 'measurement mask' : 'transcriptome image'}`;
      img.width = 384; img.height = 384; img.loading = 'lazy'; img.decoding = 'async';
      imageBox.append(img,node('span','tile-index',cell.id.replace('pretrain_','#')),node('span','tile-zoom','↗'));
      const meta = node('div','tile-meta'), label = node('div');
      label.append(node('strong',null,cell.cell_type.replaceAll('_',' ')),node('small',null,speciesName(cell.species)));
      meta.append(label,node('span',null,`${number(cell.detected_genes)} ${t('detected','检出基因')}`));
      tile.append(imageBox,meta); tile.addEventListener('click',() => openCell(cell)); fragment.append(tile);
    });
    grid.replaceChildren(fragment);
    more.hidden = species !== 'all' || expanded;
    document.querySelector('.gallery-section').classList.toggle('showing-masks',mask.checked);
    const legend = document.querySelectorAll('.image-legend span');
    legend[0].textContent = mask.checked ? t('MEASUREMENT','测量状态') : t('EXPRESSION','表达量');
    legend[1].textContent = mask.checked ? t('Unmeasured → measured','未测量 → 已测量') : t('Low → high','低 → 高');
  }
  document.querySelectorAll('.filter').forEach(button => button.addEventListener('click',() => {
    species = button.dataset.species; expanded = false;
    document.querySelectorAll('.filter').forEach(filter => {
      const active = filter === button; filter.classList.toggle('active',active); filter.setAttribute('aria-pressed',String(active));
    });
    renderGallery();
  }));
  mask.addEventListener('change',() => { if(cells.length) renderGallery(); });
  more.addEventListener('click',() => { expanded = true; renderGallery(); const tiles=grid.querySelectorAll('.cell-tile'); if(tiles[12]) tiles[12].focus({preventScroll:true}); });
  function openCell(cell) {
    currentCell = cell; linear.checked = false; updateDialog(); dialog.showModal();
    document.body.style.overflow = 'hidden';
  }
  function updateDialog() {
    const cell = currentCell;
    document.getElementById('dialog-title').textContent = `${speciesName(cell.species)} · ${cell.cell_type.replaceAll('_',' ')}`;
    const image = document.getElementById('dialog-expression'); image.src = linear.checked ? cell.linear_image : cell.image;
    image.alt = `${speciesName(cell.species)} ${cell.cell_type} expression (${linear.checked ? 'linear' : 'gamma 0.25'})`;
    document.getElementById('dialog-mask').src = cell.mask_image;
    const download = document.getElementById('dialog-download'); download.href = image.src;
    const metadata = document.getElementById('dialog-metadata'); metadata.replaceChildren();
    [[t('Cell','细胞'),cell.id],[t('Measured genes','已测基因'),number(cell.measured_genes)],[t('Detected genes','检出基因'),number(cell.detected_genes)],['Layout','384 × 384']].forEach(([label,value]) => {
      const item = node('span'); item.append(node('b',null,label),document.createTextNode(value)); metadata.append(item);
    });
  }
  linear.addEventListener('change',updateDialog);
  document.getElementById('dialog-close').addEventListener('click',() => dialog.close());
  dialog.addEventListener('click',event => { if(event.target === dialog){ const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom) dialog.close(); } });
  dialog.addEventListener('close',() => { document.body.style.overflow = ''; currentCell = null; });
  const researchTabs = [...document.querySelectorAll('.research-tab')];
  function activateTab(tab, focus) {
    researchTabs.forEach(button => { const active=button===tab;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;document.getElementById(`panel-${button.dataset.panel}`).hidden=!active; });
    if(focus) tab.focus();
  }
  researchTabs.forEach((button,index) => {
    button.addEventListener('click',() => activateTab(button,false));
    button.addEventListener('keydown',event => {
      let next;
      if(event.key==='ArrowRight') next=(index+1)%researchTabs.length;
      if(event.key==='ArrowLeft') next=(index+researchTabs.length-1)%researchTabs.length;
      if(event.key==='Home') next=0;
      if(event.key==='End') next=researchTabs.length-1;
      if(next!==undefined){event.preventDefault();activateTab(researchTabs[next],true);}
    });
  });
  let toastTimer;
  async function copyText(value) {
    try {
      if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
      else {
        const input=node('textarea'); input.value=value;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();const copied=document.execCommand('copy');input.remove();if(!copied) throw Error('Clipboard unavailable');
      }
      const toast=document.getElementById('toast');toast.textContent=t('Copied to clipboard','已复制到剪贴板');toast.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(() => {toast.hidden=true;},2400);
    } catch {
      const toast=document.getElementById('toast');toast.textContent=t('Copy unavailable. Select and copy the text.','无法自动复制，请选中文本后复制。');toast.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(() => {toast.hidden=true;},4000);
    }
  }
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click',() => copyText(button.dataset.copy)));
  document.getElementById('copy-install').addEventListener('click',() => copyText(document.getElementById('install-code').textContent));
  applyLanguage();
  fetch('assets/gallery.json').then(response => {if(!response.ok)throw Error('Gallery unavailable');return response.json();}).then(data => {cells=data.cells;renderGallery();}).catch(() => {
    grid.replaceChildren(node('p','gallery-loading',t('Images could not load. The real examples are available in Resources below.','图像暂时无法加载，可在下方资源区下载真实示例。')));
  });
})();
