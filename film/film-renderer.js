/** CellNet research film. Deterministic conceptual motion, 1920 × 1080.
 * Animated mosaics, cells and molecular modules are explanatory graphics.
 * The small dark image planes are unmodified examples from the Image Atlas.
 * Playback UI, accessible transcript and reduced-motion controls live outside
 * this renderer. No measured result plots or numerical scores are synthesized.
 */
export const DURATION = 60;
export const SCENES = [
  {id:'panorama',start:0,end:6,title:{en:'Panoramic transcriptome modelling',zh:'转录组的全景建模'},subtitle:{en:'Coding and non-coding RNA in one cellular context.',zh:'将编码与非编码 RNA 纳入共同的细胞语境。'}},
  {id:'representation',start:6,end:15,title:{en:'DNA identity meets RNA expression',zh:'DNA 身份与 RNA 表达相遇'},subtitle:{en:'DNA-derived gene identities organize measured RNA expression into a fixed, species-specific image.',zh:'DNA 衍生的基因身份组织实测 RNA 表达，形成物种内固定的图像布局。'}},
  {id:'learning',start:15,end:24,title:{en:'Cell context. Gene detail.',zh:'细胞全景，基因细节'},subtitle:{en:'Masked views learn consistent cellular context; gene-conditioned decoding preserves molecular information.',zh:'掩码视图学习一致的细胞表征，基因条件解码保留分子层面的信息。'}},
  {id:'scale',start:24,end:30,title:{en:'Learning across species',zh:'跨物种学习'},subtitle:{en:'A diverse collection of transcriptome images trains a shared foundation model.',zh:'多样的转录组图像共同训练共享的基础模型。'}},
  {id:'cells',start:30,end:38,title:{en:'Cell identity. Cell state.',zh:'细胞身份与细胞状态'},subtitle:{en:'Preserves cell identities and resolves rare populations and condition-associated states in evaluated cohorts.',zh:'在评估队列中保留细胞身份，识别稀有群体与条件相关状态。'}},
  {id:'prediction',start:38,end:47,title:{en:'Predict gene expression',zh:'预测基因表达'},subtitle:{en:'Observed context supports withheld coding and selected non-coding targets, with extensions across species and spatial measurements.',zh:'已观测语境支持留出的编码及部分非编码靶标预测，并拓展至跨物种与空间测量。'}},
  {id:'programmes',start:47,end:55,title:{en:'Molecular programmes',zh:'分子程序'},subtitle:{en:'Cell-contextual gene representations retrieve known modules and developmental or neural programmes, including from sparse non-coding RNA queries.',zh:'细胞语境中的基因表征可检索已知模块及发育、神经相关程序，也支持低检出率非编码 RNA 查询。'}},
  {id:'closing',start:55,end:60,title:{en:'CellNet: A panoramic view of the cell',zh:'CellNet：细胞的全景视角'},subtitle:{en:'Connecting cellular diversity with molecular detail.',zh:'连接细胞多样性与分子细节。'}},
];
const C={paper:'#fafbf8',ink:'#14333b',teal:'#287e7b',muted:'#69817f',line:'#d8e5de',white:'#ffffff',sage:'#eaf1eb',coral:'#ca8877',gold:'#bda474'};
const W=1920,H=1080,TAU=Math.PI*2;
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>{x=clamp(x);return x*x*(3-2*x)};
const ease=(t,start=0,duration=1)=>smooth((t-start)/duration);
const frac=x=>x-Math.floor(x);
const rand=(x,y=0)=>frac(Math.sin(x*127.1+y*311.7)*43758.5453);
const palette=['#d5e7dd','#b5d5c7','#8fbdae','#5e9f94','#35867f','#1d6468','#c7b38a'];
const ZH=lang=>lang==='zh'||lang==='zh-CN';
function load(path){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(`Cannot load film asset: ${path}`));im.src=new URL(`../assets/${path}`,import.meta.url).href})}

export async function createFilm(canvas){
  const ctx=canvas.getContext('2d',{alpha:false});
  if(!ctx)throw new Error('Canvas 2D is unavailable.');
  const assets=Object.fromEntries(await Promise.all(Object.entries({logo:'cellnet-logo.png',mouse:'cells/pretrain_0004.webp',fish:'cells/pretrain_0010.webp',fly:'cells/pretrain_0020.webp'}).map(async([key,path])=>[key,await load(path)])));
  const images=[assets.mouse,assets.fish,assets.fly];
  let language='en';
  const l=(en,zh)=>ZH(language)?zh:en;
  const family=serif=>ZH(language)?'"Noto Sans SC", "Microsoft YaHei", sans-serif':serif?'"Instrument Serif", Georgia, serif':'"DM Sans", Arial, sans-serif';
  function text(s,x,y,size=40,color=C.ink,weight=400,{align='left',serif=false,max=Infinity}={}){ctx.save();ctx.font=`${weight} ${size}px ${family(serif)}`;const width=ctx.measureText(s).width;if(width>max)ctx.font=`${weight} ${size*max/width}px ${family(serif)}`;ctx.fillStyle=color;ctx.textBaseline='alphabetic';ctx.textAlign=align;ctx.fillText(s,x,y);ctx.restore()}
  function alpha(a,draw){ctx.save();ctx.globalAlpha*=clamp(a);draw();ctx.restore()}
  function rect(x,y,w,h,fill,r=0,stroke=null,lineWidth=1){ctx.beginPath();r?ctx.roundRect(x,y,w,h,r):ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke()}}
  function line(x1,y1,x2,y2,color=C.line,width=2){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
  function circle(x,y,r,fill,stroke=null,width=1){ctx.beginPath();ctx.arc(x,y,r,0,TAU);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}
  function shadow(draw){ctx.save();ctx.shadowColor='rgba(29,63,52,.10)';ctx.shadowBlur=32;ctx.shadowOffsetY=16;draw();ctx.restore()}
  function base(t){const gradient=ctx.createLinearGradient(0,0,W,H);gradient.addColorStop(0,C.paper);gradient.addColorStop(.65,'#f8faf6');gradient.addColorStop(1,'#eaf2eb');rect(0,0,W,H,gradient);alpha(.22,()=>{for(let i=0;i<7;i++){ctx.beginPath();ctx.moveTo(1100+i*120,-50);ctx.bezierCurveTo(900+i*132,300,1600+i*90,700,1450+i*118,1150);ctx.strokeStyle='#dce8df';ctx.lineWidth=1;ctx.stroke()}})}
  function headline(s,sub='',{x=100,y=158,size=82}={}){text(s,x,y,size,C.ink,500,{max:1720});if(sub)text(sub,x,y+76,37,C.muted,400,{max:1680})}
  function label(s,x,y,{color=C.teal,align='left',size=31}={}){text(s,x,y,size,color,500,{align,max:780})}
  function imagePlane(im,x,y,size,rotation=0,opacity=1){alpha(opacity,()=>{ctx.save();ctx.translate(x,y);ctx.rotate(rotation);shadow(()=>rect(-size/2-6,-size/2-6,size+12,size+12,'#fff',8));ctx.save();ctx.beginPath();ctx.roundRect(-size/2,-size/2,size,size,4);ctx.clip();ctx.drawImage(im,-size/2,-size/2,size,size);ctx.restore();ctx.restore()})}
  function bezierPoint(points,p){const q=1-p,[a,b,c,d]=points;return{x:q*q*q*a.x+3*q*q*p*b.x+3*q*p*p*c.x+p*p*p*d.x,y:q*q*q*a.y+3*q*q*p*b.y+3*q*p*p*c.y+p*p*p*d.y}}
  function path(points,{color=C.line,width=2,alphaValue=1}={}){alpha(alphaValue,()=>{ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);ctx.bezierCurveTo(points[1].x,points[1].y,points[2].x,points[2].y,points[3].x,points[3].y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()})}
  function stream(points,t,{color=C.teal,count=5,speed=.13,offset=0,size=9,lineOpacity=.65}={}){path(points,{alphaValue:lineOpacity});for(let i=0;i<count;i++){const p=frac(t*speed+i/count+offset),q=bezierPoint(points,p),a=Math.sin(p*Math.PI);alpha(a*.8,()=>rect(q.x-size/2,q.y-size/2,size,size,color,2))}}
  function mosaic(x,y,w,h,t,{cols=30,rows=24,rotation=0,curve=0,assembly=1,origin=null,mask=0,maskPhase=0,light=.2,layer=false,outline=false,observed=false,predict=0}={}){
    ctx.save();ctx.translate(x+w/2,y+h/2);ctx.rotate(rotation);ctx.translate(-w/2,-h/2);
    const dx=w/cols,dy=h/rows,gap=Math.max(1.1,dx*.09);
    if(layer){shadow(()=>rect(-10,-10,w+20,h+20,'rgba(255,255,255,.76)',5));}
    for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
      const u=(col+.5)/cols,v=(row+.5)/rows,identity=rand(col+8,row+11),patch=rand(Math.floor(col/4)+maskPhase,Math.floor(row/4)+2);
      const isTarget=rand(col+63,row+17)>.78;
      const hidden=observed?isTarget:patch<mask;
      const settle=smooth(clamp(assembly*1.7-(u*.55+v*.15)));
      const wave=Math.max(0,Math.cos((u*.8+v*.25-t*.095)*TAU));
      const colorIndex=Math.floor(identity*(palette.length-.001));
      let xx=col*dx,yy=row*dy+curve*Math.sin(u*Math.PI);
      if(origin&&settle<1){xx=lerp(origin.x-x,xx,settle);yy=lerp(origin.y-y,yy,settle);}
      const predicted=isTarget&&predict>rand(col*3,row*7);
      const visible=hidden&&!predicted?.48:lerp(.58,.95,clamp(identity*.5+wave*light));
      const fill=hidden&&!predicted?'#e5ebe5':predicted?C.coral:palette[colorIndex];
      alpha(visible*lerp(.3,1,settle),()=>{rect(xx,yy,dx-gap,dy-gap,fill,Math.min(1.5,dx*.07));});
      if(predicted&&outline)rect(xx,yy,dx-gap,dy-gap,null,1,C.coral,1.5);
    }
    ctx.restore();
  }
  function bands(x,y,width,t,{rows=4,color=C.teal,identity=false}={}){for(let r=0;r<rows;r++){line(x,y+r*35,x+width,y+r*35,C.line,1);for(let i=0;i<22;i++){const p=frac(i/22+t*.04),xx=x+p*width,h=identity?12:7+rand(i,r)*24;alpha(Math.sin(p*Math.PI)*.9,()=>rect(xx,y+r*35-h/2,width/32,h,identity?palette[(i+r)%6]:r%2?C.coral:color,2));}}}
  function layerStack(x,y,w,h,t){for(let k=0;k<4;k++){const xx=x+k*34,yy=y-k*22;alpha(.55+.1*k,()=>{shadow(()=>rect(xx,yy,w,h,'#f3f8f3',9,C.line,1.5));for(let r=0;r<6;r++)for(let c=0;c<3;c++){const pulse=.35+.65*(.5+.5*Math.sin(t*1.8-r*.7-c*.6-k*.4));alpha(pulse,()=>rect(xx+24+c*34,yy+33+r*38,20,20,palette[(r+c+k)%6],3))}})}}
  function cellGlyph(x,y,size,color,phase=0,filled=false){ctx.save();ctx.translate(x,y);ctx.scale(1+.025*Math.sin(phase),1-.025*Math.sin(phase));circle(0,0,size,filled?color:'#fbfcf9',color,1.8);circle(-size*.12,size*.08,size*.36,filled?'rgba(255,255,255,.7)':color);ctx.restore()}
  function logo(x,y,w){ctx.save();ctx.globalCompositeOperation='multiply';ctx.drawImage(assets.logo,x,y,w,w*assets.logo.height/assets.logo.width);ctx.restore()}

  function opening(t){
    const q=t/6;
    text('CellNet',100,157,59,C.ink,600);
    text(l('Panoramic','转录组的'),100,390,103,C.ink,500,{max:1000});
    text(l('transcriptome','全景建模'),100,513,103,C.ink,500,{max:1000});
    if(!ZH(language))text('modelling',100,641,112,C.teal,400,{serif:true});
    text(l('Coding and non-coding RNA','编码与非编码 RNA'),104,ZH(language)?650:760,40,C.muted,400,{max:850});
    const px=1090-35*q,py=224-24*q;
    for(let k=2;k>=0;k--)alpha(.22+k*.08,()=>mosaic(px+30*k,py+25*k,640,610,t+k,{rotation:-.055,curve:40,assembly:1,light:.5}));
    mosaic(px,py,640,610,t,{rotation:-.055,curve:40,assembly:.74+.26*q,origin:{x:965,y:900},light:.85,layer:true});
    const flow=[{x:520,y:902},{x:790,y:878},{x:906,y:904},{x:1100,y:770}];
    stream(flow,t,{count:12,speed:.09,size:10,color:C.teal});
    const warm=[{x:760,y:975},{x:900,y:925},{x:1030,y:1010},{x:1370,y:845}];
    stream(warm,t,{count:7,speed:.1,size:8,color:C.coral});
  }

  function representation(t){
    headline(l('DNA identity meets RNA expression','DNA 身份与 RNA 表达相遇'),l('Coding and non-coding, in one image.','编码与非编码，共同构成一幅图像。'));
    const grid={x:1160,y:350,w:570,h:520};
    label(l('Identity','身份'),110,395);bands(110,440,425,t,{rows:4,identity:true});
    label(l('Expression','表达'),110,709);bands(110,760,425,t,{rows:4,color:C.coral});
    const origin={x:819,y:647};
    const upper=[{x:570,y:490},{x:765,y:490},{x:750,y:615},origin];
    const lower=[{x:570,y:802},{x:738,y:802},{x:748,y:647},origin];
    stream(upper,t,{count:7,color:C.teal,size:12,speed:.13});stream(lower,t,{count:7,color:C.coral,size:10,speed:.13,offset:.5});
    circle(origin.x,origin.y,56,'#f2f7f0',C.line,1.5);
    for(let i=0;i<5;i++){const h=15+20*(.5+.5*Math.sin(t*1.1+i));rect(origin.x-31+i*14,origin.y-h/2,8,h,i%2?C.coral:C.teal,2)}
    stream([origin,{x:950,y:647},{x:1010,y:610},{x:grid.x,y:610}],t,{count:8,size:12,speed:.14});
    // Gene positions settle once and remain fixed; only expression/visibility changes.
    mosaic(grid.x,grid.y,grid.w,grid.h,t,{assembly:.28+.72*t/9,origin,curve:0,light:.9,layer:true});
    for(let i=0;i<3;i++){const phase=frac(t*.105+i*.29),x=1030+phase*560,y=958-100*Math.sin(phase*Math.PI);alpha(Math.sin(phase*Math.PI)*.65,()=>rect(x,y,12,12,i%2?C.coral:C.teal,2))}
  }

  function learning(t){
    headline(l('Cell context. Gene detail.','细胞全景，基因细节'),l('Learn the cell. Query a gene.','学习细胞，查询基因。'));
    const inputX=130,inputY=383,s=195;
    for(let i=0;i<4;i++){
      const xx=inputX+(i%2)*243,yy=inputY+Math.floor(i/2)*268;
      const phase=Math.floor((t+i*.67)/2.6);
      mosaic(xx,yy,s,s,t,{cols:16,rows:16,mask:i<2?.18:.48,maskPhase:phase*4+i,light:.55,layer:true});
      const py=yy+s/2;
      stream([{x:xx+s+14,y:py},{x:715,y:py},{x:660,y:574},{x:822,y:574}],t,{count:4,speed:.18,offset:i*.2,size:7});
    }
    layerStack(840,441,145,296,t);
    // Parallel aligned streams represent latent agreement, not spatial crops.
    for(let k=0;k<3;k++)stream([{x:1115,y:470+k*48},{x:1220,y:470+k*48},{x:1242,y:402},{x:1380,y:402}],t,{count:5,offset:k*.17,speed:.18,color:palette[3+k],size:9});
    label(l('Cell context','细胞语境'),1388,337);
    for(let i=0;i<14;i++){const h=34+34*(.5+.5*Math.sin(t*.8-i*.28));rect(1389+i*24,402-h/2,16,h,palette[i%6],3)}
    label(l('Gene query','基因查询'),1301,918);
    bands(1301,840,260,t,{rows:1,identity:true});
    stream([{x:1570,y:840},{x:1690,y:840},{x:1653,y:676},{x:1590,y:653}],t,{count:5,speed:.17,color:C.coral,size:10});
    stream([{x:1555,y:448},{x:1555,y:500},{x:1542,y:541},{x:1542,y:598}],t,{count:4,speed:.17,size:9});
    for(let k=0;k<3;k++){rect(1435+k*25,589-k*16,151,106,'#eef5ed',5,C.line,1.5);for(let j=0;j<4;j++)alpha(.4+.5*Math.sin((t*.7+j)*Math.PI)**2,()=>rect(1455+k*25+j*25,620-k*16,12,42,palette[(j+k)%6],2))}
    stream([{x:1640,y:653},{x:1740,y:653},{x:1740,y:567},{x:1797,y:567}],t,{count:4,speed:.17,color:C.coral,size:8});
    for(let i=0;i<4;i++){const height=24+44*(.5+.5*Math.sin(t*1.1-i*.4));rect(1773+i*15,537-height/2,8,height,i%2?C.coral:C.teal,2)}
  }

  function scale(t){
    headline(l('Learning across species','跨物种学习'),l('Diverse cells. Shared representations.','多样的细胞，共享的表征。'));
    // These are actual Image Atlas image textures, repeated as a conceptual flow.
    for(let row=0;row<4;row++)for(let i=0;i<7;i++){
      const p=frac(i/7+t*.071+row*.061),startY=365+row*159;
      const x=lerp(-80,1348,p),y=lerp(startY,608,smooth(Math.max(0,(p-.56)/.44)));
      const size=lerp(84+rand(i,row)*75,40,smooth(Math.max(0,(p-.6)/.4)));
      const opacity=Math.sin(p*Math.PI)*.81;
      imagePlane(images[(i+row)%3],x,y,size,lerp(-.045,.025,rand(i,row)),opacity);
    }
    for(let j=0;j<4;j++)stream([{x:900,y:380+j*145},{x:1180,y:380+j*145},{x:1180,y:608},{x:1342,y:608}],t,{count:6,speed:.14,size:7,lineOpacity:.35,color:palette[j+1]});
    mosaic(1357,422,361,361,t,{cols:22,rows:22,light:.85,layer:true});
    rect(1409,553,257,99,'rgba(250,252,248,.94)',6);text('CellNet',1538,619,55,C.ink,500,{align:'center'});
    label('Image Atlas',100,963,{size:34});label('Foundation Model',1538,904,{align:'center',size:34});
  }

  function cells(t){
    headline(l('Cell identity. Cell state.','细胞身份与细胞状态'),l('From broad populations to fine differences.','从广泛群体，到细微状态差异。'));
    const groups=[{x:489,y:591,n:38,r:205,c:'#4c9390'},{x:1115,y:557,n:30,r:193,c:'#83aaa8'},{x:1544,y:776,n:9,r:99,c:C.coral}];
    for(let k=0;k<groups.length;k++){
      const g=groups[k];
      // Cell glyphs (membrane + nucleus), no plot axes or measured embeddings.
      for(let i=0;i<g.n;i++){
        const a=i*2.39996,rr=Math.sqrt((i+.5)/g.n)*g.r;
        const initial={x:140+rand(i,k)*1570,y:385+rand(i+8,k+1)*490};
        const form=smooth(clamp((t+.8-i*.055)/5.7));
        const stateShift=k===1?46*Math.sin(t*.48+i*.25):k===0?19*Math.sin(t*.32+i*.2):8*Math.sin(t*.5+i);
        const xx=lerp(initial.x,g.x+Math.cos(a)*rr,form)+stateShift;
        const yy=lerp(initial.y,g.y+Math.sin(a)*rr,form)+8*Math.cos(t*.55+i*.7);
        const activation=(.5+.5*Math.sin(t*.7-i*.2-k))>.82;
        cellGlyph(xx,yy,13+rand(i,k)*9,g.c,t+i,activation);
      }
    }
    // Fine-state changes are a coherent sweep through a community, not point jitter.
    const sweep=frac(t*.13),sx=938+sweep*400;
    alpha(.12,()=>{const g=ctx.createLinearGradient(sx-60,0,sx+60,0);g.addColorStop(0,'rgba(73,142,138,0)');g.addColorStop(.5,'rgba(73,142,138,.35)');g.addColorStop(1,'rgba(73,142,138,0)');rect(sx-60,330,120,460,g)});
    label(l('Rare populations','稀有群体'),1544,968,{align:'center',color:C.coral});
  }

  function prediction(t){
    headline(l('Predict gene expression','预测基因表达'),t<4.7?l('Coding and selected non-coding targets.','编码与部分非编码靶标。'):l('Across species and spatial measurements.','拓展至跨物种与空间测量。'));
    const ox=110,oy=401,ow=549,oh=494;
    mosaic(ox,oy,ow,oh,t,{cols:25,rows:23,observed:true,light:.6,layer:true});
    label(l('Observed context','已观测语境'),110,354);
    const queryY=886;
    bands(820,queryY,250,t,{rows:1,identity:true});
    label(l('Gene query','基因查询'),945,969,{align:'center'});
    layerStack(807,465,111,236,t);
    stream([{x:684,y:640},{x:734,y:640},{x:734,y:578},{x:791,y:578}],t,{count:6,speed:.2,size:10});
    // The compact decoder is separate from the encoder: only it receives the query.
    for(let k=0;k<3;k++){rect(1090+k*14,595-k*12,86,98,'#f1f6ef',5,C.line,1.4);for(let j=0;j<3;j++)rect(1103+k*14+j*22,622-k*12,11,33,palette[(j+k)%6],2)}
    stream([{x:1030,y:578},{x:1065,y:578},{x:1057,y:638},{x:1082,y:638}],t,{count:4,speed:.2,size:8});
    stream([{x:1069,y:queryY},{x:1148,y:queryY},{x:1148,y:745},{x:1148,y:702}],t,{count:6,speed:.18,size:10,color:C.coral});
    stream([{x:1212,y:638},{x:1227,y:638},{x:1230,y:640},{x:1251,y:640}],t,{count:3,speed:.19,size:8,color:C.coral});
    // Target slots are unobserved on the left; coral outlines identify predicted
    // slots on the right. The query identity travels only into decoding.
    mosaic(1283,401,537,494,t,{cols:25,rows:23,observed:true,predict:clamp((t+.2)/8.2),outline:true,light:.6,layer:true});
    // Sequential target queries continue during the whole scene.
    const col=Math.floor(frac(t*.19)*25),row=Math.floor(frac(t*.11+.17)*23),dx=537/25,dy=494/23;
    rect(1283+col*dx-2,401+row*dy-2,dx+3,dy+3,null,2,C.coral,3);
  }

  function programmes(t){
    headline(l('Molecular programmes','分子程序'),l('From sparse RNA queries to biological context.','从低检出率 RNA 查询，连接生物学语境。'));
    const q={x:390,y:610};
    const ring=62+11*Math.sin(t*.55);
    circle(q.x,q.y,ring+18,null,'#d2e2d9',1.5);circle(q.x,q.y,ring,'#e1eee5',C.teal,2);
    for(let i=0;i<5;i++){const hh=19+20*(.5+.5*Math.sin(t*.7+i));rect(q.x-30+i*14,q.y-hh/2,8,hh,i%2?C.coral:C.teal,2)}
    label('SERTAD4-AS1',q.x,762,{align:'center',size:35});
    const modules=[{x:1105,y:470,n:20,r:130},{x:1557,y:570,n:24,r:156},{x:1165,y:845,n:16,r:115}];
    for(let k=0;k<modules.length;k++){
      const m=modules[k];
      const connector=[{x:q.x+ring+18,y:q.y},{x:750,y:q.y},{x:810,y:m.y},{x:m.x-m.r,y:m.y}];
      // Plain lines indicate retrieval relations, never regulatory direction.
      path(connector,{color:'#9ebeb0',width:2,alphaValue:.3+.7*ease(t,k*.65,2)});
      for(let i=0;i<m.n;i++){
        const phase=clamp((t-.2-k*.42-i*.14)/1.7),a=i*2.39996,rr=Math.sqrt((i+.5)/m.n)*m.r;
        const end={x:m.x+Math.cos(a)*rr,y:m.y+Math.sin(a)*rr};
        const move=smooth(phase),x=lerp(m.x-m.r,end.x,move),y=lerp(m.y,end.y,move);
        alpha(.12+.88*move,()=>{if(i>0){const prev=(i-1)*2.39996,pr=Math.sqrt((i-.5)/m.n)*m.r;line(x,y,m.x+Math.cos(prev)*pr,m.y+Math.sin(prev)*pr,'#c7dad0',1.5)}const s=12+rand(i,k)*9;rect(x-s/2,y-s/2,s,s,palette[(i+k*2)%6],3);});
      }
      const phase=frac(t*.14+k*.21),p=bezierPoint(connector,phase);
      alpha(Math.sin(phase*Math.PI)*.65,()=>circle(p.x,p.y,5,k===1?C.coral:C.teal));
    }
    label(l('Neural programme','神经程序'),1450,973,{align:'center',size:35});
  }

  function closing(t){
    const q=t/5;
    logo(80,60,405);
    text('CellNet:',100,398,112,C.ink,500);
    text(l('A panoramic view','细胞的'),100,528,103,C.ink,400,{max:1060});
    text(l('of the cell','全景视角'),100,665,121,C.teal,400,{serif:!ZH(language)});
    text('daydayup-xq.github.io/CellNet',105,828,38,C.ink,500,{max:1020});
    const x=1220-24*q,y=310-22*q;
    mosaic(x+31,y+24,507,486,t+55,{cols:26,rows:25,rotation:-.045,curve:23,light:.7});
    mosaic(x,y,507,486,t+55,{cols:26,rows:25,rotation:-.045,curve:23,light:.7,layer:true});
    const points=[{x:780,y:944},{x:1015,y:962},{x:1100,y:935},{x:1270,y:819}];
    stream(points,t,{count:12,speed:.075,size:8,color:C.teal,lineOpacity:.4});
  }

  const draw=[opening,representation,learning,scale,cells,prediction,programmes,closing];
  function render(time,lang='en'){
    time=clamp(Number.isFinite(time)?time:0,0,DURATION);language=lang;
    let i=SCENES.findIndex(s=>time>=s.start&&time<s.end);if(i<0)i=SCENES.length-1;
    const scene=SCENES[i],t=time-scene.start;
    ctx.save();ctx.setTransform(canvas.width/W,0,0,canvas.height/H,0,0);base(time);
    const a=i>0?ease(t,0,.65):1;
    if(a<1)alpha(1-a,()=>draw[i-1](SCENES[i-1].end-SCENES[i-1].start+t));
    alpha(a,()=>draw[i](t));ctx.restore();return{time,scene:scene.id,index:i};
  }
  render(0);return{render,assets};
}
