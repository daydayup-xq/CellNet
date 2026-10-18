/** CellNet — 36 seconds at 120 BPM. One English-language visual master.
 * Gene/cell diagrams are conceptual; atlas textures are actual CellNet images.
 * A complete panorama opens the film; no image splitting or whole-frame fades.
 */
export const DURATION=36;
export const BPM=120;
export const BEAT_MARKERS=Array.from({length:73},(_,i)=>i/2);
const scene=(id,start,end,title,subtitle)=>({id,start,end,title:{en:title,zh:title},subtitle:{en:subtitle,zh:subtitle}});
export const SCENES=[
 scene('panorama',0,4,'Panoramic transcriptome modelling','CellNet brings coding and non-coding RNA into one panoramic cellular context.'),
 scene('representation',4,8,'DNA expands the gene vocabulary.','DNA-derived identities bring coding and non-coding genes into a fixed, species-specific expression image. One selected scRNA-seq cell row supplies the expression values; measured zeros and unmeasured genes remain distinct.'),
 scene('atlas',8,14,'An atlas for what comes next.','An open collection of transcriptome images, connecting diverse cells and species in a common visual representation.'),
 scene('learning',14,18,'Panoramic biology. Practical compute.','Image patches combine many genes into a bounded token grid, making panoramic transcriptome modelling practical at scale. Full-extent masked views preserve cellular context.'),
 scene('cells',18,22,'See the rare. Resolve the state.','CellNet preserves identities and resolves rare populations and condition-associated states in evaluated cohorts.'),
 scene('prediction',22,26,'Explore beyond measured genes.','From observed cellular context and separate DNA-derived gene queries, CellNet predicts withheld coding and selected non-coding expression, extending exploration across species and spatial measurement panels.'),
 scene('programmes',26,30,'Non-coding RNA. Functional connections.','Non-coding RNA queries retrieve associated molecular modules and developmental or neural gene programmes, including from sparsely detected queries.'),
 scene('closing',30,36,'Image Atlas + Foundation Model','A broader foundation for biological discovery.'),
];
const W=1920,H=1080,TAU=Math.PI*2;
const C={paper:'#fafbf8',ink:'#14333b',teal:'#287e7b',muted:'#708781',line:'#d6e3db',coral:'#c98673'};
const palette=['#d0e3d7','#aecdbd','#83b5a1','#599e8e','#318177','#1d6266','#c5b184'];
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),lerp=(a,b,p)=>a+(b-a)*p;
const ease=v=>{v=clamp(v);return v*v*v*(v*(v*6-15)+10)};
const phase=(t,a,b)=>ease((t-a)/(b-a)),frac=v=>v-Math.floor(v);
const rand=(x,y=0)=>frac(Math.sin(x*127.1+y*311.7)*43758.5453);
const COLS=30,ROWS=24;
const genes=Array.from({length:COLS*ROWS},(_,id)=>({id,col:id%COLS,row:Math.floor(id/COLS),color:palette[Math.floor(rand(id,14)*palette.length)],hidden:rand(id+63,17)>.79}));
// The conversion table is a schematic cell × gene matrix, not measured assay values.
// Every visible column has a stable gene ID; all pixels use the same selected cell.
const tableGeneIds=[31,86,143,199,242,311,358,409,520];
const tableColumnById=new Map(tableGeneIds.map((id,col)=>[id,col]));
const expressionForGene=id=>({missing:id%13===0,zero:id%13!==0&&id%11===0,value:.25+.65*rand(id,27)});
const arrivalForGene=id=>tableColumnById.has(id)?5.5+Math.floor(tableColumnById.get(id)/2)*.5:5.5+Math.floor(rand(id,9)*5)*.5;
const blend=(a,b,p)=>({x:lerp(a.x,b.x,p),y:lerp(a.y,b.y,p),size:lerp(a.size,b.size,p)});
function load(path){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(`Cannot load film asset: ${path}`));im.src=new URL(`../assets/${path}`,import.meta.url).href})}
export async function createFilm(canvas){
 const ctx=canvas.getContext('2d',{alpha:false});if(!ctx)throw new Error('Canvas 2D unavailable.');
 const entries=[['logo','cellnet-logo.png'],...Array.from({length:24},(_,i)=>[`cell${i}`,`cells/pretrain_${String(i).padStart(4,'0')}.webp`])];
 const assets=Object.fromEntries(await Promise.all(entries.map(async([key,path])=>[key,await load(path)])));
 const images=Array.from({length:24},(_,i)=>assets[`cell${i}`]);
 function rect(x,y,w,h,fill,r=0,stroke=null,width=1){if(w<=0||h<=0)return;ctx.beginPath();r?ctx.roundRect(x,y,w,h,Math.min(r,w/2,h/2)):ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}
 function circle(x,y,r,fill,stroke=null,width=1){if(r<=0)return;ctx.beginPath();ctx.arc(x,y,r,0,TAU);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}
 function line(a,b,color=C.line,width=2){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
 function text(value,x,y,size=40,{color=C.ink,serif=false,align='left',max=1720}={}){ctx.save();const family=serif?'"Instrument Serif", Georgia, serif':'"DM Sans", Arial, sans-serif';ctx.font=`500 ${size}px ${family}`;const w=ctx.measureText(value).width;if(w>max)ctx.font=`500 ${size*max/w}px ${family}`;ctx.textAlign=align;ctx.textBaseline='alphabetic';ctx.fillStyle=color;ctx.fillText(value,x,y);ctx.restore()}
 function shadow(draw){ctx.save();ctx.shadowColor='rgba(24,62,48,.09)';ctx.shadowBlur=22;ctx.shadowOffsetY=10;draw();ctx.restore()}
 function pathPoint(p,t){const q=1-t,[a,b,c,d]=p;return{x:q*q*q*a.x+3*q*q*t*b.x+3*q*t*t*c.x+t*t*t*d.x,y:q*q*q*a.y+3*q*q*t*b.y+3*q*t*t*c.y+t*t*t*d.y}}
 function path(points,color=C.line,width=2,progress=1){const a=pathPoint(points,0);ctx.beginPath();ctx.moveTo(a.x,a.y);for(let k=1;k<=30;k++){const p=pathPoint(points,progress*k/30);ctx.lineTo(p.x,p.y)}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
 function flow(points,t,{color=C.teal,count=5,speed=.5,size=8,progress=1}={}){path(points,C.line,2,progress);for(let k=0;k<count;k++){const q=frac(t*speed+k/count);if(q>progress)continue;const p=pathPoint(points,q);rect(p.x-size/2,p.y-size/2,size,size,color,2)}}
 function background(){const g=ctx.createLinearGradient(0,0,W,H);g.addColorStop(0,'#fdfdfb');g.addColorStop(1,'#edf3ec');rect(0,0,W,H,g)}
 function gridPose(g,cx,cy,size,rotation=0){const x=((g.col+.5)/COLS-.5)*size,y=((g.row+.5)/ROWS-.5)*size;return{x:cx+x*Math.cos(rotation)-y*Math.sin(rotation),y:cy+x*Math.sin(rotation)+y*Math.cos(rotation),size:size/COLS*.9}}
 function panorama(t){if(t>=8)return;const pull=phase(t,0,3.5),turn=phase(t,3.5,4.5),size=lerp(735,620,pull),x=lerp(1390,1320,pull),y=lerp(562,600,pull);ctx.save();ctx.translate(x,y);ctx.rotate(lerp(-.115,0,turn));shadow(()=>rect(-size/2-14,-size/2-14,size+28,size+28,'#f7faf4',10,C.line,1.2));
  for(const g of genes){const p=gridPose(g,0,0,size),left=p.x-p.size/2,top=p.y-p.size/2;
   if(t<4){rect(left,top,p.size,p.size,g.color,1.2);if(g.id%23===0){const h=(.45+.25*Math.sin(t*Math.PI-g.row*.24))*p.size;rect(left,p.y+p.size/2-h,p.size,h,C.coral,1)}}
   else{
    // DNA-defined positions stay fixed as one cell's RNA populates the same canvas.
    const prepare=phase(t,4,4.5),state=expressionForGene(g.id),arrive=phase(t,arrivalForGene(g.id),arrivalForGene(g.id)+.16);
    rect(left,top,p.size,p.size,'#e5ece3',1.2);
    rect(left,top,p.size,p.size*lerp(1,.17,prepare),g.color,1.1);
    if(state.missing&&prepare===1)line({x:p.x-p.size*.16,y:p.y+2},{x:p.x+p.size*.16,y:p.y+2},'#a9b9af',1.2);
    else if(!state.missing&&arrive>0){
     if(state.zero)rect(left+p.size*.2,top+p.size*.38,p.size*.6,p.size*.42,null,.5,C.teal,1.2*arrive);
     else{const h=p.size*.72*state.value*arrive;rect(left+1,top+p.size-1-h,p.size-2,h,C.coral,.7)}
    }
   }
  }ctx.restore();
 }
 function conversion(t){if(t<4||t>=8)return;
  const reveal=phase(t,4,4.35),select=phase(t,4.5,4.8),tableX=104,tableY=596,tableW=694,tableH=248,rowY=696;
  ctx.save();ctx.beginPath();ctx.rect(tableX-8,tableY-8,tableW+16,(tableH+16)*reveal);ctx.clip();
  shadow(()=>rect(tableX,tableY,tableW,tableH,'#f9fbf7',10,C.line,1.3));
  if(select>0)rect(tableX+10,rowY-21,tableW-20,42,'#e3eee4',6,C.teal,1.6*select);
  const drawValue=(state,cx,cy,size)=>{
   if(state.missing)line({x:cx-size*.24,y:cy},{x:cx+size*.24,y:cy},'#a4b5aa',2);
   else if(state.zero)rect(cx-size*.22,cy-size*.22,size*.44,size*.44,null,1,C.teal,1.7);
   else{const h=size*state.value;rect(cx-size*.25,cy+size*.44-h,size*.5,h,C.coral,1.5)}
  };
  for(let row=0;row<4;row++){
   const cy=650+row*46;text(`Cell ${row+1}`,tableX+19,cy+8,24,{color:row===1?C.teal:C.muted,max:100});
   for(let col=0;col<tableGeneIds.length;col++){
    const id=tableGeneIds[col],state=row===1?expressionForGene(id):{missing:(id+row*3)%13===0,zero:(id+row)%11===0,value:.2+.7*rand(id,row+40)},cx=225+col*58;
    drawValue(state,cx,cy,29);
    if(row===0){text(`G${col+1}`,cx,621,23,{align:'center',color:C.muted,max:48});rect(cx-12,627,24,3,genes[id].color,1)}
   }
   text('…',747,cy+7,27,{align:'center',color:C.muted,max:40});
  }
  ctx.restore();
  // Only the highlighted row emits values. A missing gene never emits a marker.
  for(let col=0;col<tableGeneIds.length;col++){
   const id=tableGeneIds[col],g=genes[id],state=expressionForGene(id);if(state.missing)continue;
   const arrival=arrivalForGene(id),departure=arrival-.5,p=clamp((t-departure)/.5),target=gridPose(g,1320,600,620);
   if(t>=departure&&t<arrival){
    const points=[{x:225+col*58,y:rowY},{x:884,y:rowY},{x:target.x-80,y:target.y},{x:target.x,y:target.y}];
    path(points,'#cfddd2',1.7);const pos=pathPoint(points,ease(p)),size=lerp(27,target.size,p);
    circle(pos.x,pos.y,size*.78,'#f7faf3');
    if(state.zero)rect(pos.x-size*.26,pos.y-size*.26,size*.52,size*.52,null,1,C.teal,2);
    else{const h=size*state.value;rect(pos.x-size*.3,pos.y+size*.4-h,size*.6,h,C.coral,1)}
   }
   const landed=phase(t,arrival,arrival+.12)*(1-phase(t,arrival+.12,arrival+.42));
   if(landed>0)rect(target.x-target.size*.8,target.y-target.size*.8,target.size*1.6,target.size*1.6,null,2,C.teal,2*landed);
  }
  if(t>=4.5){
   rect(106,903,15,15,null,1,C.teal,1.8);text('Measured zero',139,920,27,{color:C.muted,max:260});
   line({x:410,y:911},{x:426,y:911},'#a4b5aa',2.2);text('Unmeasured',443,920,27,{color:C.muted,max:300});
  }
 }
 function imageCard(index,x,y,size,framed=true){if(size<.5)return;ctx.save();ctx.translate(x,y);if(framed)shadow(()=>rect(-size/2-5,-size/2-5,size+10,size+10,'#fff',4));ctx.drawImage(images[index===9?4:index===4?9:index%24],-size/2,-size/2,size,size);ctx.restore()}
 function atlasPose(index,t){const col=index%6,row=Math.floor(index/6),distance=Math.abs(col-3)+Math.abs(row-1),start=8+Math.max(0,distance-1)*.5,p=phase(t,start,start+.8),main=index===9;let size=main?lerp(620,155,phase(t,8,10)):155*p,x=lerp(1320,860+col*175,p)-phase(t,10,13.5)*22,y=lerp(600,355+row*175,p)-phase(t,10,13.5)*48;if(t>=13.5){const a=13.5+((col+row)%6)*.25,q=phase(t,a,a+.8);x=lerp(x,1335,q);y=lerp(y,583,q);size=lerp(size,0,q)}return{x,y,size}}
 function atlas(t){if(t<8||t>17)return;const order=Array.from({length:24},(_,i)=>i).filter(i=>i!==9);order.push(9);for(const i of order){const p=atlasPose(i,t);imageCard(i,p.x,p.y,p.size,t<14.5);if(t>=10&&t<13.5&&i===Math.floor((t-10)*2)*3)rect(p.x-p.size/2-6,p.y-p.size/2-6,p.size+12,p.size+12,null,4,C.coral,2.5)}}
 function grid(cx,cy,size,t,{masked=false,variant=0,neutralTargets=false,prediction=0}={}){for(const g of genes){const p=gridPose(g,cx,cy,size),hidden=neutralTargets&&g.hidden||masked&&rand(Math.floor(g.col/5)+variant,Math.floor(g.row/4))<.38,revealed=neutralTargets&&g.hidden&&prediction>rand(g.id,31);rect(p.x-p.size/2,p.y-p.size/2,p.size,p.size,hidden?(revealed?C.coral:'#dfe9df'):g.color,1);if(revealed)rect(p.x-p.size/2,p.y-p.size/2,p.size,p.size,null,1,C.coral,1.1)}}
 function modelLayers(x,y,scale,t,{closing=false}={}){ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);for(let k=0;k<4;k++){const dx=-105+k*34,dy=-170-k*15;shadow(()=>rect(dx,dy,175,340,'#f1f6ef',9,'#c9dbcf',1.5));for(let r=0;r<6;r++)for(let c=0;c<3;c++){const pulse=.6+.4*Math.sin((t-r*.125-k*.25)*Math.PI*2);rect(dx+25+c*43,dy+40+r*44,20,12+20*pulse,palette[(c+r+k)%6],2)}}ctx.restore();if(!closing)text('CellNet',x+32*scale,y+240*scale,32,{align:'center',color:C.teal,max:300})}
 function patchGroups(cx,cy,size,t,start){
  // These schematic patch outlines group existing gene pixels into a bounded grid.
  // They never crop the panorama or reveal values beneath a masked input slot.
  for(let row=0;row<4;row++)for(let col=0;col<5;col++){
   const reveal=phase(t,start+col*.125,start+col*.125+.28);if(reveal===0)continue;
   rect(cx-size/2+col*size/5+2,cy-size/2+row*size/4+2,size/5-4,size/4-4,null,2,'#518d7b',1.2*reveal);
  }
 }
 function learning(t){if(t<13.5||t>=18)return;const enter=phase(t,13.5,14.5),exit=phase(t,17,18),cx=1338,cy=600;modelLayers(cx,cy,enter*(1-exit*.5),t);
  // Equal-sized, aligned full-image views converge together; masking keeps the full extent.
  const fold=phase(t,16.5,17.7);
  for(let i=0;i<3;i++){
   const start=15+i*.5,show=phase(t,start,start+.3),size=lerp(184,18,fold)*show,x=lerp(980,cx,fold),y=lerp(390+i*210,cy,fold);
   if(fold<1){
    shadow(()=>rect(x-size/2-5,y-size/2-5,size+10,size+10,'#fff',4));
    grid(x,y,size,t,{masked:true,variant:i});patchGroups(x,y,size,t,start+.1);
    const portY=cy+(i-1)*148*(1-fold);
    flow([{x:x+size/2+10,y},{x:x+size/2+60,y},{x:cx-155,y:portY},{x:cx-100,y:portY}],t,{count:3,size:6,progress:show});
   }
  }
 }
 const communities=[{x:1065,y:550,n:28,r:168,color:C.teal},{x:1530,y:480,n:29,r:168,color:'#7fae9f'},{x:1480,y:825,n:8,r:87,color:C.coral}],selected={x:1108,y:620};
 function cellScene(t){if(t<17.5||t>=22.5)return;const enter=phase(t,17.5,19),exit=phase(t,21.5,22.5);for(let k=0;k<3;k++){const group=communities[k],grow=phase(t,18+k*.5,19+k*.5);for(let i=0;i<group.n;i++){const a=i*2.39996,r=Math.sqrt((i+.5)/group.n)*group.r,dx=Math.cos(a)*r,dy=Math.sin(a)*r;let x=lerp(1350,group.x+dx,enter)+dx*exit*10,y=lerp(600,group.y+dy,enter)+dy*exit*10;const radius=(12+rand(i,k)*7)*grow;circle(x,y,radius,C.paper,group.color,1.8);circle(x-2,y+2,radius*.34,group.color)}if(k===2&&t>19.5)circle(group.x,group.y,lerp(45,124,phase(t,19.5,20.5)),null,'#cda998',1.5)}if(t<21.5){circle(selected.x,selected.y,47*enter,C.paper,C.teal,1.6);grid(selected.x,selected.y,49*enter,t)}}
 function selectedPanorama(t){if(t<21.5||t>=26)return;const p=phase(t,21.5,22.5),cx=lerp(selected.x,382,p),cy=lerp(selected.y,640,p),size=lerp(49,480,p);shadow(()=>rect(cx-size/2-8,cy-size/2-8,size+16,size+16,'#fff',6));grid(cx,cy,size,t,{neutralTargets:true})}
 function decoder(x,y,s=1){ctx.save();ctx.translate(x,y);ctx.scale(s,s);for(let k=0;k<3;k++){rect(k*15,-k*12,96,114,'#eef4eb',6,C.line,1.4);for(let j=0;j<3;j++)rect(17+k*15+j*24,31-k*12,12,42,palette[(j+k)%6],2)}ctx.restore()}
 function prediction(t){if(t<22||t>=26)return;const enter=phase(t,22,22.5),outputX=1518,outputY=640,size=480*enter;modelLayers(868,640,.65*enter,t);shadow(()=>rect(outputX-size/2-8,outputY-size/2-8,size+16,size+16,'#fff',6));const beat=Math.floor(Math.max(0,t-23)*2),p=clamp((beat+phase(t,23+beat*.5,23.3+beat*.5))/5);grid(outputX,outputY,size,t,{neutralTargets:true,prediction:p});decoder(1089,596,enter);
  if(enter>.5){flow([{x:636,y:640},{x:700,y:640},{x:708,y:640},{x:774,y:640}],t,{count:4});flow([{x:998,y:640},{x:1024,y:640},{x:1040,y:640},{x:1084,y:640}],t,{count:3});flow([{x:1228,y:640},{x:1260,y:640},{x:1238,y:640},{x:1267,y:640}],t,{color:C.coral,count:3});
   // DNA query enters only the separate decoder. Input hidden targets stay neutral.
   for(let j=0;j<11;j++)rect(880+j*19,911,12,14+rand(j,33)*24,palette[j%6],2);
   flow([{x:1095,y:925},{x:1155,y:925},{x:1155,y:790},{x:1155,y:714}],t,{color:C.coral,count:4});text('Observed context',145,947,31,{color:C.muted,max:490});text('DNA query',881,1000,30,{color:C.teal,max:300});text('Predicted expression',1282,947,31,{color:C.muted,max:500});
  }
 }
 function modulePosition(g,t){const group=g.id%3,n=Math.floor(g.id/3),centers=[{x:1110,y:430},{x:1530,y:610},{x:1110,y:823}],center=centers[group],r=25+Math.sqrt(n/240)*(group===1?155:136),a=n*2.39996;return blend(gridPose(g,1518,640,480),{x:center.x+Math.cos(a)*r,y:center.y+Math.sin(a)*r,size:g.id%5===0?9:3},phase(t,26,27.5))}
 function programmes(t){if(t<26||t>=30.7)return;const enter=phase(t,26,27),leave=phase(t,29.8,30.7);ctx.save();ctx.translate(0,-1100*leave);const query={x:400,y:663};circle(query.x,query.y,56*enter,'#e2eee5',C.teal,1.5);for(let j=0;j<5;j++)rect(query.x-32+j*14,query.y-17,8,34*enter,j%2?C.coral:C.teal,2);
  for(let group=0;group<3;group++){const center=[{x:1110,y:430},{x:1530,y:610},{x:1110,y:823}][group],grow=phase(t,26.5+group*.5,27.5+group*.5),points=[{x:456,y:663},{x:740,y:663},{x:790,y:center.y},{x:center.x-95,y:center.y}];path(points,'#b3cdbd',2,grow);const q=frac((t-26)*.5+group/3);if(q<grow){const p=pathPoint(points,q);rect(p.x-4,p.y-4,8,8,C.teal,2)}}
  for(let i=0;i<genes.length;i+=15){const a=modulePosition(genes[i],t),b=modulePosition(genes[(i+45)%genes.length],t),grow=phase(t,27+Math.floor((i%90)/30)*.5,28+Math.floor((i%90)/30)*.5);line(a,{x:lerp(a.x,b.x,grow),y:lerp(a.y,b.y,grow)},'#c5d9ca',1.3)}for(const g of genes){const p=modulePosition(g,t);rect(p.x-p.size/2,p.y-p.size/2,p.size,p.size,g.color,2)}
  if(t>=27){text('ncRNA query',query.x,query.y+119,32,{color:C.teal,align:'center',max:450});text('Development',1110,245,30,{color:C.muted,align:'center',max:400});text('Neural programmes',1518,838,30,{color:C.muted,align:'center',max:430})}ctx.restore();
 }
 function closing(t){if(t<30)return;const enter=phase(t,30,31),y=lerp(1250,564,enter),cardY=y-223;
  shadow(()=>rect(104,cardY,804,440,'#edf4ed',14,C.line,1.3));shadow(()=>rect(1012,cardY,804,440,'#eef4f0',14,C.line,1.3));
  for(let i=0;i<9;i++){const col=i%3,row=Math.floor(i/3),delay=31+(i%3)*.5,p=phase(t,delay,delay+.45),x=lerp(506,352+col*153,p),yy=lerp(y,y-145+row*145,p);imageCard(i*2,x,yy,lerp(3,128,p),false)}
  const modelX=1525,modelY=y+15,inputX=1164,inputSize=96,columnCenterY=cardY+220;
  modelLayers(modelX,modelY,.82*enter,t,{closing:true});
  // Identical squares, one vertical centerline, uniform 20-pixel gaps; all inside the card.
  for(let i=0;i<3;i++){
   const yy=columnCenterY+(i-1)*116;
   imageCard(10+i,inputX,yy,inputSize,false);
   if(t>31){const endX=modelX-105*.82-6;flow([{x:inputX+inputSize/2,y:yy},{x:1278,y:yy},{x:1370,y:yy},{x:endX,y:yy}],t,{count:2,size:5})}
  }
  ctx.save();ctx.globalCompositeOperation='multiply';ctx.drawImage(assets.logo,96,45,488,488*assets.logo.height/assets.logo.width);ctx.restore();if(t>=31){text('Image Atlas',130,866,75,{max:780});text('Foundation Model',1038,866,75,{max:780});text('Open for discovery.',134,925,30,{color:C.teal,max:760});text('Built for discovery.',1042,925,30,{color:C.teal,max:760})}
 }
 function label(value,x,y,size,t,start,end,{color=C.ink,serif=false,max=1720}={}){const p=phase(t,start,start+.28),q=phase(t,end-.2,end);ctx.save();ctx.beginPath();ctx.rect(x-4,y-size*1.12,max+8,size*1.36);ctx.clip();text(value,x+24*(1-p),y+size*1.35*(1-p)-size*1.35*q,size,{color,serif,max});ctx.restore()}
 function titles(t){
  if(t<4){label('CellNet',96,264,147,t,-1,4,{max:930});label('Panoramic',102,422,113,t,-1,4,{color:C.teal,serif:true,max:870});label('transcriptome modelling',103,512,65,t,-1,4,{max:900});label('A wider view of life.',105,725,39,t,1,4,{color:C.muted,max:800})}
  if(t>=4&&t<8){label('DNA expands',96,272,111,t,4,8,{max:900});label('the gene vocabulary.',96,402,96,t,4.5,8,{color:C.teal,serif:true,max:850});label('Coding and non-coding genes.',104,478,31,t,4.5,8,{color:C.muted,max:820});label('scRNA-seq',104,562,34,t,4,8,{color:C.teal,max:700});label('DNA-defined gene positions',1010,242,32,t,4.5,8,{color:C.muted,max:690});label('One cell. One image.',1010,978,32,t,5,8,{color:C.teal,max:690})}
  if(t>=8&&t<14){label('Image Atlas',98,220,54,t,8,14,{color:C.teal,max:650});label('An atlas for',96,382,101,t,8,14,{max:710});label('what comes',96,497,101,t,8.5,14,{max:710});label('next.',96,633,129,t,9,14,{color:C.teal,serif:true,max:710});label('Large-scale. Cross-species.',102,798,31,t,10,14,{color:C.muted,max:690});label('Open for discovery.',102,851,31,t,10.5,14,{color:C.teal,max:690})}
  if(t>=14&&t<18){label('Foundation Model',98,220,48,t,14,18,{color:C.teal,max:740});label('Panoramic',96,380,108,t,14,18,{max:760});label('biology.',96,506,110,t,14.5,18,{max:760});label('Practical compute.',96,631,88,t,15,18,{color:C.teal,serif:true,max:765});label('Many genes per patch.',102,794,32,t,15.5,18,{color:C.muted,max:740});label('One cellular context.',102,847,32,t,16,18,{color:C.teal,max:740})}
  if(t>=18&&t<22){label('See the rare.',96,320,104,t,18,22,{max:835});label('Resolve',96,460,112,t,18.5,22,{max:820});label('the state.',96,592,122,t,19,22,{color:C.teal,serif:true,max:820})}
  if(t>=22&&t<26){label('Explore beyond measured genes.',96,177,100,t,22,26,{max:1730});label('Predict expression across a wider gene space.',101,260,39,t,22.5,26,{color:C.teal,max:1720})}
  if(t>=26&&t<30){label('Non-coding RNA.',96,296,96,t,26,30,{max:850});label('Functional',96,430,108,t,26.5,30,{max:850});label('connections.',96,562,118,t,27,30,{color:C.teal,serif:true,max:870});label('Retrieve associated gene programmes.',102,925,31,t,27.5,30,{color:C.muted,max:820})}
  if(t>=30){label('Panoramic transcriptome',955,122,54,t,30,38,{max:865});label('modelling, at scale.',955,186,54,t,30.5,38,{color:C.teal,max:865});label('A broader foundation for biological discovery.',103,1006,45,t,32,38,{max:1710})}
 }
 function render(time,_language='en'){time=clamp(Number.isFinite(time)?time:0,0,DURATION);let index=SCENES.findIndex(s=>time>=s.start&&time<s.end);if(index<0)index=SCENES.length-1;ctx.save();ctx.setTransform(canvas.width/W,0,0,canvas.height/H,0,0);background();panorama(time);conversion(time);atlas(time);learning(time);cellScene(time);selectedPanorama(time);prediction(time);programmes(time);closing(time);titles(time);ctx.restore();return{time,scene:SCENES[index].id,index,beat:Math.min(72,Math.floor(time*2))}}
 render(0);return{render,assets};
}
