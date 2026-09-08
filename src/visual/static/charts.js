export const regionColors=['#a5e385','#53cbb9','#6db9f1','#efc56e','#db8bba'];
const stops=[[32,38,67],[40,120,135],[117,200,176],[216,232,151],[250,197,89],[216,114,73]];
export function pressureColor(t){const z=Math.max(0,Math.min(.9999,t))*(stops.length-1),i=Math.floor(z),a=z-i;return stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*a));}
export function heatmap(canvas,pressure,boxes=null,annotation=null,center=null){
 const ctx=canvas.getContext('2d'),max=Math.max(1,...pressure.flat()),small=document.createElement('canvas');small.width=24;small.height=44;
 const c=small.getContext('2d'),img=c.createImageData(24,44);
 pressure.flat().forEach((v,i)=>{const color=pressureColor(v/max);img.data.set([...color,255],i*4);});c.putImageData(img,0,0);
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=true;ctx.drawImage(small,0,0,canvas.width,canvas.height);
 const sx=canvas.width/24,sy=canvas.height/44;
 for(const [set,dashed] of [[annotation,true],[boxes,false]]){if(!set)continue;ctx.setLineDash(dashed?[5,4]:[]);set.forEach((b,i)=>{ctx.strokeStyle=regionColors[i];ctx.lineWidth=dashed?1.5:2.5;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);if(!dashed){ctx.font='11px sans-serif';ctx.fillStyle=regionColors[i];ctx.fillText(['肩','背','腰','臀','腿'][i],b[0]*sx+3,b[1]*sy+13);}});}
 ctx.setLineDash([]);if(center){const [x,y]=[(center[0]+.5)*sx,(center[1]+.5)*sy];ctx.strokeStyle='#ffffff';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();}
}
export function trendChart(svg,history,bagId){
 let content='';for(let y=15;y<=130;y+=38)content+=`<line x1="0" y1="${y}" x2="680" y2="${y}" stroke="#edf1e8" stroke-dasharray="3 5"/>`;
 const series=[history.map(h=>h.metrics.maximum),history.map(h=>h.airbags.find(b=>b.id===bagId)?.pressure??0),history.map(h=>(h.airbags.find(b=>b.id===bagId)?.fill??.5)*100)];
 series.forEach((values,k)=>{const max=Math.max(...values,1),min=0;const pts=values.map((v,i)=>`${(i/Math.max(values.length-1,1)*674+3).toFixed(1)},${(130-(v-min)/(max-min)*106).toFixed(1)}`).join(' ');content+=`<polyline points="${pts}" fill="none" stroke="${['#3e826d','#8ea964','#d0a16c'][k]}" stroke-width="2" stroke-linejoin="round" ${k===2?'stroke-dasharray="4 4"':''}/>`;});
 if(history.length<2)content+='<text x="340" y="75" text-anchor="middle" fill="#879780" font-size="11">开始回放以查看压力变化</text>';svg.innerHTML=content;
}
export function scatter(svg,points){
 if(!points.length){svg.innerHTML='<text x="340" y="200" text-anchor="middle">训练后显示用户特征</text>';return;}
 const ns='http://www.w3.org/2000/svg',xs=points.map(p=>p.x),ys=points.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),names=[...new Set(points.map(p=>p.person))];
 svg.innerHTML='';for(let i=0;i<6;i++){const line=document.createElementNS(ns,'line');Object.entries({x1:35,y1:35+i*69,x2:650,y2:35+i*69,stroke:'#edf1e7'}).forEach(([k,v])=>line.setAttribute(k,v));svg.append(line);}
 points.forEach(p=>{const x=40+(p.x-xmin)/Math.max(xmax-xmin,1)*600,y=380-(p.y-ymin)/Math.max(ymax-ymin,1)*330,node=document.createElementNS(ns,p.known?'circle':'path');if(p.known){node.setAttribute('cx',x);node.setAttribute('cy',y);node.setAttribute('r',3.5);}else node.setAttribute('d',`M${x} ${y-4}l4 4 -4 4 -4 -4Z`);node.setAttribute('fill',p.known?`hsl(${names.indexOf(p.person)*137.5%360} 35% 52%)`:'none');node.setAttribute('stroke',p.known?'none':'#ba874e');node.setAttribute('opacity','.8');const title=document.createElementNS(ns,'title');title.textContent=`${p.person} · ${p.known?'已注册':'未注册'} · 识别为 ${p.prediction} · 距离 ${p.distance.toFixed(2)}`;node.append(title);svg.append(node);});
 const label=document.createElementNS(ns,'text');label.setAttribute('x',325);label.setAttribute('y',420);label.setAttribute('fill','#8b9785');label.setAttribute('font-size','11');label.textContent='第一主成分 PC1';svg.append(label);
 const yLabel=document.createElementNS(ns,'text');yLabel.setAttribute('x',8);yLabel.setAttribute('y',20);yLabel.setAttribute('fill','#8b9785');yLabel.setAttribute('font-size','11');yLabel.textContent='PC2';svg.append(yLabel);
}
