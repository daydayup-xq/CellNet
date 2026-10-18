/** CellNet — a continuous, deterministic 40-second research film.
 * A persistent set of schematic gene tokens is followed from identity/expression
 * to a panorama, encoding, cellular context, querying and molecular programmes.
 * There are no scene-level opacity transitions, cuts or synthesized data plots.
 * Scientific diagrams are conceptual; image textures are actual Image Atlas cells.
 */
export const DURATION=40;
export const SCENES=[
 {id:'panorama',start:0,end:4,title:{en:'Panoramic transcriptome modelling',zh:'转录组的全景建模'},subtitle:{en:'CellNet brings coding and non-coding RNA into one cellular context.',zh:'CellNet 将编码与非编码 RNA 纳入共同的细胞语境。'}},
 {id:'representation',start:4,end:12,title:{en:'DNA identity meets RNA expression',zh:'DNA 身份与 RNA 表达相遇'},subtitle:{en:'DNA-derived identities organize measured RNA expression into a fixed, species-specific image.',zh:'DNA 衍生的基因身份组织实测 RNA 表达，形成物种内固定的图像布局。'}},
 {id:'learning',start:12,end:18,title:{en:'Learn the cell. Across species.',zh:'学习细胞，跨越物种'},subtitle:{en:'Full-extent masked views learn consistent cellular context. Diverse transcriptome images train a shared foundation model.',zh:'保持全图范围的掩码视图学习一致的细胞语境，多样的转录组图像共同训练共享基础模型。'}},
 {id:'cells',start:18,end:22,title:{en:'Cell identity. Cell state.',zh:'细胞身份与细胞状态'},subtitle:{en:'CellNet preserves identities and resolves rare populations and condition-associated states in evaluated cohorts.',zh:'CellNet 在评估队列中保留细胞身份，识别稀有群体与条件相关状态。'}},
 {id:'prediction',start:22,end:30,title:{en:'Query a gene',zh:'查询基因'},subtitle:{en:'Observed context supports withheld coding and selected non-coding targets, with extensions across species and spatial measurements.',zh:'已观测语境支持留出的编码及部分非编码靶标预测，并拓展至跨物种与空间测量。'}},
 {id:'programmes',start:30,end:36,title:{en:'Molecular programmes',zh:'分子程序'},subtitle:{en:'Cell-contextual gene representations retrieve known modules and developmental or neural programmes, including from sparse non-coding RNA queries.',zh:'细胞语境中的基因表征可检索已知模块及发育、神经相关程序，也支持低检出率非编码 RNA 查询。'}},
 {id:'closing',start:36,end:40,title:{en:'CellNet: A panoramic view of the cell',zh:'CellNet：细胞的全景视角'},subtitle:{en:'Connecting cellular diversity with molecular detail.',zh:'连接细胞多样性与分子细节。'}},
];
const W=1920,H=1080,TAU=Math.PI*2;
const C={paper:'#fafbf8',ink:'#14333b',teal:'#287e7b',muted:'#6b817e',line:'#d6e3db',coral:'#c98673',white:'#ffffff'};
const palette=['#cfe3d7','#a9cebc','#7fb39f','#559b8c','#318177','#1d6266','#c8b184'];
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smoother=x=>{x=clamp(x);return x*x*x*(x*(x*6-15)+10)};
const phase=(t,a,b)=>smoother((t-a)/(b-a));
const frac=x=>x-Math.floor(x);
const random=(x,y=0)=>frac(Math.sin(x*127.1+y*311.7)*43758.5453);
const zh=lang=>lang==='zh'||lang==='zh-CN';
const target=i=>random(i+63,17)>.79;
const COLS=30,ROWS=24,N=COLS*ROWS;
const genes=Array.from({length:N},(_,id)=>({id,col:id%COLS,row:Math.floor(id/COLS),color:palette[Math.floor(random(id,14)*palette.length)],target:target(id)}));
const stops=[0,4.1,8.8,11.2,13.6,15.6,18.1,21.5,24.5,29.0,32.3,35.3,38.0,40];
const stageNames=['hook','ribbons','panorama','panorama','views','encoded','cell','cell','observed','observed','modules','modules','closing','closing'];
function load(path){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(`Cannot load film asset: ${path}`));im.src=new URL(`../assets/${path}`,import.meta.url).href})}
function interpolateStops(t,values){let i=0;while(i<stops.length-2&&t>stops[i+1])i++;return{a:values[i],b:values[i+1],p:phase(t,stops[i],stops[i+1]),i}}
function mixPose(a,b,p){return{x:lerp(a.x,b.x,p),y:lerp(a.y,b.y,p),size:lerp(a.size,b.size,p),round:lerp(a.round||1,b.round||1,p)}}

export async function createFilm(canvas){
 const ctx=canvas.getContext('2d',{alpha:false});if(!ctx)throw new Error('Canvas 2D is unavailable.');
 const assets=Object.fromEntries(await Promise.all(Object.entries({logo:'cellnet-logo.png',mouse:'cells/pretrain_0004.webp',fish:'cells/pretrain_0010.webp',fly:'cells/pretrain_0020.webp'}).map(async([k,p])=>[k,await load(p)])));
 const images=[assets.mouse,assets.fish,assets.fly];let language='en';
 const l=(en,cn)=>zh(language)?cn:en;
 const family=serif=>zh(language)?'"Noto Sans SC", "Microsoft YaHei", sans-serif':serif?'"Instrument Serif", Georgia, serif':'"DM Sans", Arial, sans-serif';
 function text(s,x,y,size=40,color=C.ink,weight=500,{align='left',serif=false,max=Infinity}={}){ctx.save();ctx.font=`${weight} ${size}px ${family(serif)}`;const w=ctx.measureText(s).width;if(w>max)ctx.font=`${weight} ${size*max/w}px ${family(serif)}`;ctx.textAlign=align;ctx.textBaseline='alphabetic';ctx.fillStyle=color;ctx.fillText(s,x,y);ctx.restore()}
 function rect(x,y,w,h,fill,r=0,stroke=null,width=1){if(w<=0||h<=0)return;ctx.beginPath();r?ctx.roundRect(x,y,w,h,Math.min(r,w/2,h/2)):ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}
 function circle(x,y,r,fill,stroke=null,width=1){if(r<=0)return;ctx.beginPath();ctx.arc(x,y,r,0,TAU);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}
 function line(a,b,color=C.line,width=2){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
 function shadow(draw){ctx.save();ctx.shadowColor='rgba(25,62,49,.10)';ctx.shadowBlur=24;ctx.shadowOffsetY=12;draw();ctx.restore()}
 function pathPoint(points,p){const q=1-p,[a,b,c,d]=points;return{x:q*q*q*a.x+3*q*q*p*b.x+3*q*p*p*c.x+p*p*p*d.x,y:q*q*q*a.y+3*q*q*p*b.y+3*q*p*p*c.y+p*p*p*d.y}}
 function path(points,color=C.line,width=2,progress=1){ctx.beginPath();let p=pathPoint(points,0);ctx.moveTo(p.x,p.y);for(let j=1;j<=32;j++){p=pathPoint(points,progress*j/32);ctx.lineTo(p.x,p.y)}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
 function flow(points,t,{color=C.teal,count=6,speed=.25,size=9,progress=1}={}){path(points,C.line,1.7,progress);for(let j=0;j<count;j++){const q=frac(t*speed+j/count);if(q>progress)continue;const p=pathPoint(points,q);rect(p.x-size/2,p.y-size/2,size,size,color,2)}}
 function background(){const g=ctx.createLinearGradient(0,0,W,H);g.addColorStop(0,'#fcfcfa');g.addColorStop(1,'#edf4ed');rect(0,0,W,H,g)}
 function camera(t){const k=[{t:0,x:960,y:540,z:1},{t:3.7,x:1010,y:540,z:1.04},{t:8.8,x:970,y:540,z:1},{t:11.2,x:1000,y:565,z:1.035},{t:15.6,x:990,y:540,z:.99},{t:18.1,x:960,y:540,z:1},{t:21.5,x:910,y:560,z:1.025},{t:24.5,x:960,y:540,z:1},{t:29,x:1000,y:540,z:1.035},{t:32.3,x:960,y:540,z:1},{t:35.3,x:1000,y:565,z:1.025},{t:36,x:960,y:540,z:1},{t:40,x:960,y:540,z:1}];let i=0;while(i<k.length-2&&t>k[i+1].t)i++;const p=phase(t,k[i].t,k[i+1].t);return{x:lerp(k[i].x,k[i+1].x,p),y:lerp(k[i].y,k[i+1].y,p),z:lerp(k[i].z,k[i+1].z,p)}}
 function tileInGrid(g,x,y,w,h,{rotation=0,curve=0}={}){const px=(g.col+.5)/COLS*w-w/2,py=(g.row+.5)/ROWS*h-h/2+curve*Math.sin(g.col/(COLS-1)*Math.PI);return{x:x+w/2+px*Math.cos(rotation)-py*Math.sin(rotation),y:y+h/2+px*Math.sin(rotation)+py*Math.cos(rotation),size:Math.min(w/COLS,h/ROWS)*.87,round:1.2}}
 function pose(g,name){
  if(name==='hook'){const p=tileInGrid(g,1115,280,590,590,{rotation:-.10,curve:45});p.x+=Math.sin(g.row*.24)*35;return p}
  if(name==='ribbons'){const lane=g.row<12?0:1;return{x:1090+g.col*20,y:lane?725+(g.row-12)*10:335+g.row*10+28*Math.sin(g.col*.13),size:lane?8+random(g.id)*10:13,round:2}}
  if(name==='panorama')return tileInGrid(g,1135,335,610,550);
  if(name==='views')return tileInGrid(g,190,430,390,350,{rotation:-.025});
  if(name==='encoded')return{x:1190+(g.col%10)*15+(g.row%4)*11,y:452+Math.floor(g.row/4)*28+(g.col%3)*3,size:5.8,round:1};
  if(name==='cell')return tileInGrid(g,541,598,38,35);
  if(name==='observed')return tileInGrid(g,145,390,555,480);
  if(name==='modules'){const group=g.id%3,n=Math.floor(g.id/3),r=25+Math.sqrt(n/240)*(group===1?190:160),a=n*2.39996;const centers=[{x:1050,y:470},{x:1512,y:613},{x:1097,y:838}];return{x:centers[group].x+Math.cos(a)*r,y:centers[group].y+Math.sin(a)*r,size:g.id%5===0?10:3.6,round:2}}
  return tileInGrid(g,1230,355,458,428,{rotation:-.045,curve:20});
 }
 function tokenPose(g,t){const st=interpolateStops(t,stageNames);const a=pose(g,st.a),b=pose(g,st.b);let p=st.p;if(st.a==='ribbons'&&st.b==='panorama')p=smoother(clamp(st.p*1.34-(g.col/COLS)*.25-(g.row/ROWS)*.09));return mixPose(a,b,p)}
 function mainTokens(t){
  const maskWindow=phase(t,11.5,12.6)*(1-phase(t,14.7,15.6));
  const observed=phase(t,21.8,23.3)*(1-phase(t,29.3,31.2));
  for(const g of genes){const p=tokenPose(g,t);let fill=g.color;
   const hidden=(observed>.25&&g.target)||(maskWindow>.25&&random(Math.floor(g.col/4)+Math.floor(t*1.4),Math.floor(g.row/4))<.34);
   if(hidden)fill='#e3ebe3';
   // Fixed identity colors remain stable. Expression is an in-place overlay.
   rect(p.x-p.size/2,p.y-p.size/2,p.size,p.size,fill,p.round);
   if(!hidden&&t<12&&g.id%19===0){const q=.55+.45*Math.sin(t*2.3-g.row*.22);rect(p.x-p.size/2,p.y+p.size/2-p.size*q,p.size,p.size*q,g.id%2?C.coral:g.color,1)}
  }
 }
 function inputSignals(t){
  if(t>=11.8)return;
  const enter=phase(t,2.6,4.4),exit=phase(t,10.3,11.8),off=-710*(1-enter)-750*exit;
  for(let lane=0;lane<2;lane++){
   const y=lane?757:453;
   for(let row=0;row<4;row++)for(let j=0;j<23;j++){const q=frac(j/23+t*.11),x=110+q*455+off;const h=lane?7+random(j,row)*24:12;rect(x,y+row*28-h/2,12,h,lane?(row%2?C.coral:C.teal):palette[(j+row)%6],2)}
   const start={x:580+off,y:y+42},end={x:1090,y:lane?720:510};
   if(enter>0&&exit<1)flow([start,{x:770+off*.6,y:y+42},{x:825,y:lane?685:542},end],t,{color:lane?C.coral:C.teal,count:7,size:9,speed:.26,progress:enter*(1-exit)});
  }
 }
 function copyGrid(t,center,size,mask=0,rotation=0){ctx.save();ctx.translate(center.x,center.y);ctx.rotate(rotation);shadow(()=>rect(-size/2-6,-size*.45-6,size+12,size*.9+12,'#fff',5));for(const g of genes){const s=size/COLS*.87,x=-size/2+g.col*size/COLS,y=-size*.45+g.row*size*.9/ROWS;const hide=random(Math.floor(g.col/4)+Math.floor(t),Math.floor(g.row/4)+6)<mask;rect(x,y,s,s,hide?'#e3ebe3':g.color,1)}ctx.restore()}
 function viewCopies(t){if(t<10.8||t>15.8)return;const unfold=phase(t,10.8,13.1),collapse=phase(t,13.8,15.8);for(let k=0;k<3;k++){const start={x:1440,y:610},fan=[{x:470,y:357},{x:700,y:650},{x:440,y:864}][k],model={x:1280,y:570};const x=lerp(lerp(start.x,fan.x,unfold),model.x,collapse),y=lerp(lerp(start.y,fan.y,unfold),model.y,collapse),size=lerp(lerp(0,230,unfold),18,collapse);copyGrid(t+k,{x,y},size,.2+k*.15,lerp(.015,k===1?.035:-.035,unfold));}}
 function modelPose(t){if(t<10.8)return{x:1272,y:1460,s:1};if(t<15.8)return{x:1272,y:lerp(1460,586,phase(t,10.8,12.8)),s:1};if(t<21.8)return{x:lerp(1272,2160,phase(t,18.0,21.8)),y:586,s:1};if(t<24.0)return{x:2160,y:586,s:.88};return{x:lerp(2160,938,phase(t,24.0,25.7)),y:586,s:.88};}
 function model(t){if(t<10.8||t>30.8)return;const p=modelPose(t),out=phase(t,29.5,30.8);ctx.save();ctx.translate(p.x+1100*out,p.y);ctx.scale(p.s,p.s);for(let k=0;k<4;k++){const x=-125+k*30,y=-170-k*17;shadow(()=>rect(x,y,172,325,'#f1f6f0',8,C.line,1.5));for(let row=0;row<6;row++)for(let col=0;col<3;col++){const h=15+15*(.5+.5*Math.sin(t*2.1-row*.6-k));rect(x+26+col*42,y+42+row*40,19,h,palette[(row+col+k)%6],2)}}ctx.restore();}
 function speciesInputs(t){if(t<14.4||t>18.6)return;for(let lane=0;lane<3;lane++)for(let j=0;j<4;j++){const q=clamp((t-14.4-j*.22-lane*.13)/2.1),size=lerp(176,22,phase(q,.42,1)),x=lerp(-180,1320,q),y=lerp(355+lane*230,574,smoother(q));if(q===0||q===1)continue;ctx.save();ctx.translate(x,y);ctx.rotate((lane-1)*.025);shadow(()=>rect(-size/2-4,-size/2-4,size+8,size+8,'#fff',5));ctx.drawImage(images[(lane+j)%3],-size/2,-size/2,size,size);ctx.restore();}}
 const communities=[{x:560,y:615,n:24,r:177,c:'#35867f'},{x:1160,y:550,n:30,r:180,c:'#77a9a0'},{x:1576,y:805,n:7,r:88,c:C.coral}];
 function cells(t){if(t<17.2||t>25.1)return;const emerge=phase(t,17.2,20.0),zoom=phase(t,21.7,24.4),magnify=1+zoom*10,selected={x:560,y:615};for(let group=0;group<communities.length;group++){const g=communities[group];for(let j=0;j<g.n;j++){const a=j*2.39996,r=Math.sqrt((j+.5)/g.n)*g.r;let x=lerp(1280,g.x+Math.cos(a)*r,emerge),y=lerp(580,g.y+Math.sin(a)*r,emerge);x=selected.x+(x-selected.x)*magnify;y=selected.y+(y-selected.y)*magnify;const radius=(12+random(j,group)*7)*lerp(.4,1,emerge)*magnify;if(x<-200||x>W+200||y<-200||y>H+200)continue;circle(x,y,radius,C.paper,g.c,1.7);circle(x-radius*.1,y+radius*.08,radius*.33,g.c)}}
  const radius=lerp(29,1400,zoom);circle(selected.x,selected.y,radius,C.paper,C.teal,2.2);
  if(zoom<.03)circle(selected.x,selected.y,11,C.teal);
 }
 function prediction(t){if(t<23.2||t>31.2)return;const enter=phase(t,23.2,25.1),exit=phase(t,29.3,31.2),xout=1287+850*(1-enter)+1000*exit,yout=390;
  shadow(()=>rect(xout-8,yout-8,565,496,'#fff',5));
  const progress=phase(t,25.9,29.3);
  for(const g of genes){const s=555/COLS*.87,x=xout+g.col*555/COLS,y=yout+g.row*480/ROWS;const predicted=g.target&&progress>random(g.id,31);rect(x,y,s,s,g.target?(predicted?C.coral:'#e3ebe3'):g.color,1);if(predicted)rect(x,y,s,s,null,1,C.coral,1.3)}
  if(t<24.4)return;
  const decoderX=1121+850*(1-phase(t,24.4,25.8))+1000*exit;
  for(let k=0;k<3;k++){rect(decoderX+k*14,585-k*12,98,113,'#f0f5ed',5,C.line,1.4);for(let j=0;j<3;j++)rect(decoderX+17+k*14+j*24,614-k*12,12,42,palette[(j+k)%6],2)}
  const qy=916+420*(1-phase(t,24.4,25.8));
  for(let j=0;j<11;j++){const h=14+random(j,33)*23;rect(886+j*19+1000*exit,qy-h/2,12,h,palette[j%6],2)}
  const mp=modelPose(t),decoder={x:decoderX+64,y:698};
  flow([{x:718,y:630},{x:770,y:630},{x:780,y:575},{x:mp.x-142,y:575}],t,{count:5,speed:.34,size:8});
  flow([{x:mp.x+125,y:575},{x:mp.x+170,y:575},{x:decoderX-40,y:632},{x:decoderX-8,y:632}],t,{count:5,speed:.35,size:8});
  flow([{x:1098+1000*exit,y:qy},{x:decoder.x,y:qy},{x:decoder.x,y:780},decoder],t,{color:C.coral,count:6,speed:.34,size:9});
  flow([{x:decoderX+133,y:630},{x:decoderX+170,y:630},{x:xout-50,y:630},{x:xout-10,y:630}],t,{color:C.coral,count:4,speed:.35,size:9});
 }
 function programmeEdges(t){if(t<29.4||t>37.5)return;const enter=phase(t,29.4,32.6),exit=phase(t,35.3,37.5),origin={x:lerp(1040,340,enter),y:lerp(920,621,enter)};const scale=1-exit;
  circle(origin.x,origin.y,58*scale,'#e2eee5',C.teal,2);
  for(let j=0;j<5;j++)rect(origin.x-31+j*14,origin.y-16,8,32*scale,j%2?C.coral:C.teal,2);
  const cs=[{x:1050,y:470},{x:1512,y:613},{x:1097,y:838}];
  for(let k=0;k<3;k++){const center=cs[k];const p=[{x:origin.x+58*scale,y:origin.y},{x:700,y:origin.y},{x:770,y:center.y},{x:center.x-120,y:center.y}];path(p,'#9cbdae',2,enter*scale);const travel=frac(t*.38+k*.23);if(travel<enter*scale){const mark=pathPoint(p,travel);rect(mark.x-4,mark.y-4,8,8,C.teal,2)}for(let j=k;j<N;j+=30){const a=tokenPose(genes[j],t),b=tokenPose(genes[(j+45)%N],t);if(enter>.1)line(a,{x:lerp(a.x,b.x,enter*scale),y:lerp(a.y,b.y,enter*scale)},'#c2d8cb',1.4)}}
 }
 function logoReveal(t){if(t<36)return;const p=phase(t,36.3,37.8);ctx.save();ctx.beginPath();ctx.rect(74,64,590*p,236);ctx.clip();ctx.globalCompositeOperation='multiply';ctx.drawImage(assets.logo,74,64,590,590*assets.logo.height/assets.logo.width);ctx.restore();}
 function maskText(value,x,y,size,t,start,end,{color=C.ink,serif=false,max=1720}={}){const enter=phase(t,start,start+.42),leave=phase(t,end-.36,end);const width=max;ctx.save();ctx.beginPath();ctx.rect(x-4,y-size*1.03,width*(1-leave),size*1.33);ctx.clip();text(value,x+24*(1-enter),y+size*1.3*(1-enter),size,color,500,{serif,max});ctx.restore();}
 function titles(t){
  if(t<4.2){maskText('CellNet',98,270,144,t,-.5,4.2,{max:800});maskText(l('Panoramic transcriptome','转录组的全景建模'),103,404,zh(language)?75:73,t,-.4,4.2,{max:960});if(!zh(language))maskText('modelling',103,510,105,t,-.3,4.2,{color:C.teal,serif:true,max:900});}
  if(t>=4&&t<8.3)maskText('DNA × RNA',100,187,128,t,4,8.3,{max:1100});
  if(t>=7.95&&t<12.2)maskText(l('One panoramic image','一幅转录组全景图'),100,187,105,t,7.95,12.2,{max:1700});
  if(t>=11.85&&t<15.25)maskText(l('Learn the cell','学习细胞'),100,187,119,t,11.85,15.25,{max:1700});
  if(t>=14.9&&t<18.3)maskText(l('Across species','跨物种学习'),100,187,119,t,14.9,18.3,{max:1700});
  if(t>=17.95&&t<22.2)maskText(l('Identity. State.','身份与状态'),100,187,119,t,17.95,22.2,{max:1700});
  if(t>=21.85&&t<25.3)maskText(l('Observed context','已观测语境'),100,187,113,t,21.85,25.3,{max:1700});
  if(t>=24.95&&t<30.3){maskText(l('Query a gene','查询基因'),100,187,122,t,24.95,30.3,{max:1700});maskText(l('From context to expression','从语境到表达'),105,264,36,t,25.25,30.3,{color:C.muted,max:1600});}
  if(t>=29.95&&t<36.3){maskText(l('Molecular programmes','分子程序'),100,187,106,t,29.95,36.3,{max:1700});if(t>31.1)maskText('SERTAD4-AS1',217,771,37,t,31.1,36.3,{color:C.teal,max:430});}
  if(t>=36){maskText('CellNet:',100,438,111,t,36,42,{max:1010});maskText(l('A panoramic view','细胞的'),100,564,103,t,36.15,42,{max:1050});maskText(l('of the cell','全景视角'),100,694,121,t,36.3,42,{color:C.teal,serif:!zh(language),max:1050});maskText('daydayup-xq.github.io/CellNet',104,861,38,t,36.8,42,{max:1030});}
 }
 function render(time,lang='en'){
  time=clamp(Number.isFinite(time)?time:0,0,DURATION);language=lang;
  let index=SCENES.findIndex(s=>time>=s.start&&time<s.end);if(index<0)index=SCENES.length-1;
  ctx.save();ctx.setTransform(canvas.width/W,0,0,canvas.height/H,0,0);background();
  const cam=camera(time);ctx.save();ctx.translate(W/2,H/2);ctx.scale(cam.z,cam.z);ctx.translate(-cam.x,-cam.y);
  // All objects use one absolute timeline. Changes are pose/geometry changes;
  // no full-frame alpha compositing or swapping independent scene renderers.
  inputSignals(time);viewCopies(time);speciesInputs(time);cells(time);programmeEdges(time);mainTokens(time);model(time);prediction(time);
  ctx.restore();logoReveal(time);titles(time);ctx.restore();
  return{time,scene:SCENES[index].id,index};
 }
 render(0);return{render,assets};
}
