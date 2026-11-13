import sharp from 'sharp';
import {Robot} from '../robot/control.ts';
import {bodies} from '../robot/geometry.ts';
import {pose,mul,quatPose,inversePose,apply,rotate,position,add,scale,norm,sub,type V,type M} from '../robot/math.ts';

type View={transform:M;inverse:M;fx:number;fy:number;cx:number;cy:number;width:number;height:number};
type Silhouette={view:View;mask:Uint8Array;camera:string;robotDepth:Float32Array;region:any;prompt:any};
type Solid={x0:number;y0:number;spacing:number;nx:number;ny:number;low:Float32Array;high:Float32Array};
export class Vision {
 heights=new Map<string,{height:number;shape:string;fit:number}>();
 surfaces:any[]=[];
 supportZ=0.765;
 memoryKey(text:string,supportZ=this.supportZ) { return `${supportZ}:${text.trim().toLowerCase().replace(/\s+/g,' ')}`; }
 remember(text:string,result:any) {
  const step=this.robot.episode.read().step;
  this.surfaces=this.surfaces.filter(s=>s.step===step);
  if(result._surface)this.surfaces.push(result._surface);
  if(!text.trim())return;
  const key=this.memoryKey(text);
  if(result.height_memory==='rejected')this.heights.delete(key);
  if(result.found&&result.views>=2&&result.fit>=0.9&&Number.isFinite(result.height_m))this.heights.set(key,{height:result.height_m,shape:result.shape,fit:result.fit});
 }
 surfaceAt(camera:string,row:number,col:number) {
  const step=this.robot.episode.read().step;
  this.surfaces=this.surfaces.filter(s=>s.step===step);
  for(const surface of this.surfaces)if(surface.camera===camera&&surface.support_z===this.supportZ&&surface.mask[row*surface.width+col]) {
   const xyz=this.cast(this.camera(camera),surface.solid,row,col);
   if(xyz)return {xyz,views:surface.views,note:surface.note};
  }
  return null;
 }
 async backProject(camera:string,row:number,col:number,supportZ=0.765) {
  this.supportZ=supportZ;
  const view=this.camera(camera);
  if(row<0||row>=view.height||col<0||col>=view.width)throw new Error('Pixel outside the current camera');
  const cached=this.surfaceAt(camera,row,col);
  const result:any=cached?{surface_xyz:cached.xyz,views:cached.views,note:cached.note}:await this.locate(camera,{point:[row,col],min_score:0.2},{},0,supportZ);
  if(!cached)this.remember('',result);
  let xyz=result.surface_xyz??null,surface=xyz?'object':'unresolved',note=result.note??null;
  if(!xyz){
   xyz=this.rayPlane(camera,row,col,supportZ).xyz;
   if(Math.abs(xyz[0])>0.71||xyz[1]< -0.61||xyz[1]>0.51)throw new Error('Pixel looks beyond the table; no RGB depth is available');
   surface='table';note=[note,'No measurable solid at this pixel; xyz is the support-plane estimate and can represent an unresolved flat target'].filter(Boolean).join('; ');
  }
  return {pixel:[row,col],xyz,depth_m:-apply(view.inverse,xyz)[2],camera,surface,views:result.views??0,note};
 }
 sight(value:any,camera:string) {
  const result:any={found:value.found,camera,step:this.robot.episode.read().step};
  for(const key of ['center_xyz','top_z','height_m','length_m','width_m','long_axis_deg','grasp_yaw','score','centroid','note','height_memory','fit','geometry','shape','surface_xyz'])result[key]=value[key]??null;
  return {...result,area_px:value.area_px??0,views:value.views??0,view_support:value.view_support??[],source:value.source,unit:'m',support_z:this.supportZ,assumptions:value.assumptions};
 }

 boxDepth(view:View,transform:M,bounds:V[],depth:Float32Array) {
  const vertices:V[]=[];
  for(let i=0;i<8;i++)vertices.push(apply(view.inverse,apply(transform,bounds.map((range,k)=>range[(i>>k)&1]))));
  const faces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
  for(const face of faces)for(const indices of [[face[0],face[1],face[2]],[face[0],face[2],face[3]]]) {
   const points=indices.map(i=>vertices[i]);if(points.some(p=>-p[2]<=0.0001))continue;
   const pixels=points.map(p=>[view.cx+view.fx*p[0]/-p[2],view.cy-view.fy*p[1]/-p[2],-p[2]]);
   const [a,b,c]=pixels,area=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(area)<1e-9)continue;
   const x0=Math.max(0,Math.floor(Math.min(...pixels.map(p=>p[0])))),x1=Math.min(view.width-1,Math.ceil(Math.max(...pixels.map(p=>p[0])))),y0=Math.max(0,Math.floor(Math.min(...pixels.map(p=>p[1])))),y1=Math.min(view.height-1,Math.ceil(Math.max(...pixels.map(p=>p[1]))));
   for(let r=y0;r<=y1;r++)for(let col=x0;col<=x1;col++) {
    const u=((b[1]-c[1])*(col-c[0])+(c[0]-b[0])*(r-c[1]))/area,v=((c[1]-a[1])*(col-c[0])+(a[0]-c[0])*(r-c[1]))/area,w=1-u-v;
    if(u>=0&&v>=0&&w>=0){const index=r*view.width+col;depth[index]=Math.min(depth[index],1/(u/a[2]+v/b[2]+w/c[2]));}
   }
  }
 }
 robotDepth(view:View,step=-1) {
  const depth=new Float32Array(view.width*view.height);depth.fill(Infinity);
  const state=this.robot.episode.read(step).state;
  for(const side of ['left','right'] as const) {
   for(const body of bodies(this.robot.kin,side,state[side+'_arm_joint_state'],state[side+'_ee_joint_state'][0])) {
    const ranges=[0,1,2].map(index=>[body.bounds[0][index],body.bounds[1][index]]);
    this.boxDepth(view,body.transform,ranges,depth);
   }
  }
  return depth;
 }
 constructor(public robot:Robot,public url:string) {}
 camera(name:string,step=-1):View {
  const config=this.robot.kin.config;const observation=this.robot.episode.read(step);const head=name==='cam_head';
  const sensor=head?config.head_sensor:config.wrist_sensor;
  const transform=head?pose(config.head.pos,config.head.ori.map((v:number)=>v*Math.PI/180)):mul(quatPose(observation.state[(name==='cam_left_wrist'?'left':'right')+'_ee_pose']),this.robot.kin.wristMount());
  const [width,height]=sensor.resolution;const fx=width*sensor.focal_length/sensor.horizontal_aperture;
  return {transform,inverse:inversePose(transform),fx,fy:fx,cx:width/2,cy:height/2,width,height};
 }
 rayPlane(camera:string,row:number,col:number,z:number,step=-1) {
  const view=this.camera(camera,step);const direction=rotate(view.transform,[(col-view.cx)/view.fx,-(row-view.cy)/view.fy,-1]);
  const origin=position(view.transform);const distance=(z-origin[2])/direction[2];
  if(!Number.isFinite(distance)||distance<=0)throw new Error('Pixel ray does not intersect the specified plane in front of camera');
  return {xyz:add(origin,scale(direction,distance)),z,camera,source:'specified_plane'};
 }
 async segment(camera:string,query:any,signal?:AbortSignal,observation=this.robot.episode.read()) {
  const deadline=AbortSignal.timeout(90000);
  const response=await fetch(this.url+'/segment',{method:'POST',headers:{'content-type':'application/json','connection':'close'},body:JSON.stringify({image:observation.images[camera],...Object.fromEntries(['text','point','points','labels','box','min_score'].filter(k=>query[k]!==undefined).map(k=>[k,query[k]]))}),signal:signal?AbortSignal.any([signal,deadline]):deadline});
  if(!response.ok)throw new Error(`Segmentation HTTP ${response.status}: ${await response.text()}`);
  return {...await response.json() as any,camera,step:observation.step};
 }
 connectedMask(mask:Uint8Array,width:number) {
  const seen=new Uint8Array(mask.length),groups:number[][]=[];
  for(let start=0;start<mask.length;start++)if(mask[start]&&!seen[start]) {
   const group=[start];seen[start]=1;
   for(let i=0;i<group.length;i++) {
    const pixel=group[i],col=pixel%width;
    for(const next of [pixel-width,pixel+width,...(col>0?[pixel-1]:[]),...(col<width-1?[pixel+1]:[])])if(next>=0&&next<mask.length&&mask[next]&&!seen[next]){seen[next]=1;group.push(next);}
   }
   groups.push(group);
  }
  if(!groups.length)return mask;
  groups.sort((a,b)=>b.length-a.length);
  const distance=new Int16Array(mask.length);distance.fill(1000);
  for(const pixel of groups[0])distance[pixel]=0;
  for(let i=0;i<mask.length;i++)distance[i]=Math.min(distance[i],i>=width?distance[i-width]+1:1000,i%width?distance[i-1]+1:1000);
  for(let i=mask.length-1;i>=0;i--)distance[i]=Math.min(distance[i],i+width<mask.length?distance[i+width]+1:1000,i%width<width-1?distance[i+1]+1:1000);
  const kept=new Uint8Array(mask.length);
  for(const group of groups)if(group.some(pixel=>distance[pixel]<=12))for(const pixel of group)kept[pixel]=255;
  return kept;
 }
 project(view:View,point:V) {
  const [x,y,z]=apply(view.inverse,point);if(z>=0)return null;
  return [Math.round(view.cy+y/z*view.fy),Math.round(view.cx-x/z*view.fx)];
 }
 evidence(s:Silhouette,x:number,y:number,z:number) {
  const m=s.view.inverse,depth=-(m[2][0]*x+m[2][1]*y+m[2][2]*z+m[2][3]);
  if(depth<=0)return -1;
  const u=s.view.cx+s.view.fx*(m[0][0]*x+m[0][1]*y+m[0][2]*z+m[0][3])/depth,v=s.view.cy-s.view.fy*(m[1][0]*x+m[1][1]*y+m[1][2]*z+m[1][3])/depth;
  const col=Math.round(u),row=Math.round(v);
  if(row<0||col<0||row>=s.view.height||col>=s.view.width)return -1;
  const index=row*s.view.width+col;
  const c0=Math.floor(u),r0=Math.floor(v),a=u-c0,b=v-r0;
  let weight=0;for(const [dr,dy] of [[0,1-b],[1,b]])for(const [dc,dx] of [[0,1-a],[1,a]])if(r0+dr>=0&&c0+dc>=0&&r0+dr<s.view.height&&c0+dc<s.view.width)weight+=Number(s.mask[(r0+dr)*s.view.width+c0+dc]>0)*dx*dy;
  if(weight>=0.3)return 1;
  return s.robotDepth[index]<depth?-1:0;
 }
 cast(view:View,solid:Solid,row:number,col:number):V|null {
  const origin=position(view.transform),direction=rotate(view.transform,[(col-view.cx)/view.fx,-(row-view.cy)/view.fy,-1]);
  const lower=[solid.x0-solid.spacing/2,solid.y0-solid.spacing/2,this.supportZ],upper=[solid.x0+(solid.nx-0.5)*solid.spacing,solid.y0+(solid.ny-0.5)*solid.spacing,(this.supportZ+0.36)];
  let enter=0,leave=Infinity;
  for(let i=0;i<3;i++) {
   if(Math.abs(direction[i])<1e-12){if(origin[i]<lower[i]||origin[i]>upper[i])return null;continue;}
   const a=(lower[i]-origin[i])/direction[i],b=(upper[i]-origin[i])/direction[i];enter=Math.max(enter,Math.min(a,b));leave=Math.min(leave,Math.max(a,b));
  }
  if(leave<enter||leave<=0)return null;
  let t=enter+1e-9;
  let ix=Math.max(0,Math.min(solid.nx-1,Math.floor((origin[0]+t*direction[0]-lower[0])/solid.spacing))),iy=Math.max(0,Math.min(solid.ny-1,Math.floor((origin[1]+t*direction[1]-lower[1])/solid.spacing)));
  const sx=direction[0]>=0?1:-1,sy=direction[1]>=0?1:-1;
  let tx=Math.abs(direction[0])<1e-12?Infinity:(lower[0]+(ix+(sx>0?1:0))*solid.spacing-origin[0])/direction[0];
  let ty=Math.abs(direction[1])<1e-12?Infinity:(lower[1]+(iy+(sy>0?1:0))*solid.spacing-origin[1])/direction[1];
  const dx=Math.abs(solid.spacing/direction[0]),dy=Math.abs(solid.spacing/direction[1]);
  for(let n=0;n<solid.nx+solid.ny+2&&ix>=0&&iy>=0&&ix<solid.nx&&iy<solid.ny;n++) {
   const next=Math.min(tx,ty,leave),index=iy*solid.nx+ix,low=solid.low[index],high=solid.high[index];
   if(Number.isFinite(low)&&Number.isFinite(high)) {
    let start=t,end=next;
    if(Math.abs(direction[2])<1e-12){if(origin[2]<this.supportZ+low||origin[2]>this.supportZ+high)end=-1;}
    else {const a=(this.supportZ+low-origin[2])/direction[2],b=(this.supportZ+high-origin[2])/direction[2];start=Math.max(start,Math.min(a,b));end=Math.min(end,Math.max(a,b));}
    if(start<=end+1e-9)return add(origin,scale(direction,start));
   }
   if(next>=leave)break;
   if(tx<ty){t=tx;tx+=dx;ix+=sx;}else {t=ty;ty+=dy;iy+=sy;}
  }
  return null;
 }
 overlap(mask:Uint8Array,depth:Float32Array) {
  let count=0,onRobot=0;for(let i=0;i<mask.length;i++)if(mask[i]){count++;if(Number.isFinite(depth[i]))onRobot++;}
  return onRobot/Math.max(1,count);
 }
 hiddenRim(s:Silhouette) {
  const distance=new Uint8Array(s.mask.length);distance.fill(4);
  for(let i=0;i<s.mask.length;i++)if(s.mask[i])distance[i]=0;
  const width=s.view.width;
  for(let i=0;i<distance.length;i++)distance[i]=Math.min(distance[i],i>=width?distance[i-width]+1:4,i%width?distance[i-1]+1:4);
  for(let i=distance.length-1;i>=0;i--)distance[i]=Math.min(distance[i],i+width<distance.length?distance[i+width]+1:4,i%width<width-1?distance[i+1]+1:4);
  let rim=0,robot=0;for(let i=0;i<distance.length;i++)if(distance[i]>0&&distance[i]<=3){rim++;if(Number.isFinite(s.robotDepth[i]))robot++;}
  return rim>0&&robot/rim>=0.15;
 }
 coverage(solid:Solid,s:Silhouette) {
  const count=s.mask.reduce((n,v)=>n+Number(v>0),0),stride=Math.max(1,Math.floor(count/1500));
  let seen=0,hit=0,k=0;
  for(let i=0;i<s.mask.length;i++)if(s.mask[i]&&k++%stride===0){seen++;if(this.cast(s.view,solid,Math.floor(i/s.view.width),i%s.view.width))hit++;}
  return hit/Math.max(1,seen);
 }
 match(solid:Solid,s:Silhouette) {
  let r0=s.view.height,r1=0,c0=s.view.width,c1=0;
  for(let i=0;i<s.mask.length;i++)if(s.mask[i]){const r=Math.floor(i/s.view.width),c=i%s.view.width;r0=Math.min(r0,r);r1=Math.max(r1,r);c0=Math.min(c0,c);c1=Math.max(c1,c);}
  for(const x of [solid.x0,solid.x0+(solid.nx-1)*solid.spacing])for(const y of [solid.y0,solid.y0+(solid.ny-1)*solid.spacing])for(const z of [this.supportZ,(this.supportZ+0.36)]){
   const p=this.project(s.view,[x,y,z]);if(!p)continue;r0=Math.min(r0,p[0]);r1=Math.max(r1,p[0]);c0=Math.min(c0,p[1]);c1=Math.max(c1,p[1]);
  }
  r0=Math.max(0,r0);r1=Math.min(s.view.height-1,r1);c0=Math.max(0,c0);c1=Math.min(s.view.width-1,c1);
  const stride=Math.max(1,Math.ceil(Math.sqrt((r1-r0+1)*(c1-c0+1)/2500)));
  let truth=0,predicted=0,both=0;
  for(let r=r0;r<=r1;r+=stride)for(let c=c0;c<=c1;c+=stride) {
   const index=r*s.view.width+c,inside=s.mask[index]>0;
   if(!inside&&Number.isFinite(s.robotDepth[index]))continue;
   const hit=this.cast(s.view,solid,r,c)!==null;
   truth+=Number(inside);predicted+=Number(hit);both+=Number(inside&&hit);
  }
  return {fit:2*both/Math.max(1,truth+predicted),coverage:both/Math.max(1,truth)};
 }
 rectangle(points:V[]) {
  const sorted=points.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const turn=(a:V,b:V,c:V)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const half=(items:V[])=>{const stack:V[]=[];for(const p of items){while(stack.length>1&&turn(stack[stack.length-2],stack[stack.length-1],p)<=0)stack.pop();stack.push(p);}return stack;};
  const hull=[...half(sorted).slice(0,-1),...half([...sorted].reverse()).slice(0,-1)];
  let best:any=null;
  for(let i=0;i<hull.length;i++) {
   const next=hull[(i+1)%hull.length],angle=Math.atan2(next[1]-hull[i][1],next[0]-hull[i][0]),c=Math.cos(angle),s=Math.sin(angle);
   const u=hull.map(p=>p[0]*c+p[1]*s),v=hull.map(p=>-p[0]*s+p[1]*c),lo=[Math.min(...u),Math.min(...v)],hi=[Math.max(...u),Math.max(...v)];
   const width=hi[0]-lo[0],length=hi[1]-lo[1],area=width*length;
   if(!best||area<best.area)best={area,width,length,angle:angle*180/Math.PI,x:(lo[0]+hi[0])/2*c-(lo[1]+hi[1])/2*s,y:(lo[0]+hi[0])/2*s+(lo[1]+hi[1])/2*c};
  }
  return best;
 }
 geometry(points:V[]) {
  const quantile=(values:number[],fraction:number)=>{const ordered=[...values].sort((a,b)=>a-b),index=(ordered.length-1)*fraction,lo=Math.floor(index),hi=Math.ceil(index);return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo);};
  let seen=points.filter(p=>p[2]>(this.supportZ+0.004));if(seen.length<12)return null;
  const middle=[0,1].map(k=>quantile(seen.map(p=>p[k]),0.5)),distance=seen.map(p=>Math.hypot(p[0]-middle[0],p[1]-middle[1])),cutoff=Math.max(0.03,4*quantile(distance,0.5));
  seen=seen.filter((_,i)=>distance[i]<=cutoff);if(seen.length<12)return null;
  const top=quantile(seen.map(p=>p[2]),0.97),bottom=Math.min(...seen.map(p=>p[2]));
  const circles:{center:V;radius:number}[]=[];
  for(let z=(this.supportZ+0.004);z<top;z+=0.01) {
   const slice=seen.filter(p=>p[2]>=z&&p[2]<z+0.01);if(slice.length<15)continue;
   const mean=[0,1].map(k=>slice.reduce((sum,p)=>sum+p[k],0)/slice.length),xy=slice.map(p=>[p[0]-mean[0],p[1]-mean[1]]);
   let xx=0,yy=0,cross=0,bx=0,by=0,r2=0;
   for(const [x,y] of xy){const r=x*x+y*y;xx+=x*x;yy+=y*y;cross+=x*y;bx+=x*r/2;by+=y*r/2;r2+=r;}
   const determinant=xx*yy-cross*cross;if(determinant<=1e-16)continue;
   const offset=[(bx*yy-by*cross)/determinant,(by*xx-bx*cross)/determinant],center=[mean[0]+offset[0],mean[1]+offset[1]],radius=Math.sqrt(r2/slice.length+offset[0]**2+offset[1]**2);
   if(radius<0.008||radius>0.12)continue;
   const residual=Math.sqrt(slice.reduce((sum,p)=>sum+(Math.hypot(p[0]-center[0],p[1]-center[1])-radius)**2,0)/slice.length),angles=slice.map(p=>Math.atan2(p[1]-center[1],p[0]-center[0])).sort((a,b)=>a-b);
   let gap=angles[0]+2*Math.PI-angles[angles.length-1];for(let i=1;i<angles.length;i++)gap=Math.max(gap,angles[i]-angles[i-1]);
   if(residual<=radius*0.04&&2*Math.PI-gap>=100*Math.PI/180)circles.push({center,radius});
  }
  let accepted=circles;
  if(circles.length>=3){const axis=[0,1].map(k=>quantile(circles.map(c=>c.center[k]),0.5));accepted=circles.filter(c=>Math.hypot(c.center[0]-axis[0],c.center[1]-axis[1])<=0.006);if(accepted.length*2<=circles.length)accepted=[];}
  const footprint=seen.filter(p=>top-bottom<=0.012||p[2]>bottom+0.003).map(p=>p.slice(0,2));
  for(const circle of accepted)for(let a=0;a<360;a+=5)footprint.push([circle.center[0]+circle.radius*Math.cos(a*Math.PI/180),circle.center[1]+circle.radius*Math.sin(a*Math.PI/180)]);
  const cells=[...new Map(footprint.map(p=>{const x=Math.round(p[0]/0.002),y=Math.round(p[1]/0.002);return [`${x},${y}`,[x*0.002,y*0.002]];})).values()];
  if(cells.length<3)return null;
  const mean=[0,1].map(k=>cells.reduce((sum,p)=>sum+p[k],0)/cells.length);let xx=0,yy=0,cross=0;
  for(const p of cells){const x=p[0]-mean[0],y=p[1]-mean[1];xx+=x*x;yy+=y*y;cross+=x*y;}
  const spread=Math.hypot(xx-yy,2*cross),major=(xx+yy+spread)/2,minor=(xx+yy-spread)/2;
  const angle=major>=1.4*Math.max(minor,1e-12)?Math.atan2(2*cross,xx-yy)/2:this.rectangle(cells).angle*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
  const u=cells.map(p=>p[0]*c+p[1]*s),v=cells.map(p=>-p[0]*s+p[1]*c),lo=[quantile(u,0.005),quantile(v,0.005)],hi=[quantile(u,0.995),quantile(v,0.995)];
  return {x:(lo[0]+hi[0])/2*c-(lo[1]+hi[1])/2*s,y:(lo[0]+hi[0])/2*s+(lo[1]+hi[1])/2*c,width:hi[0]-lo[0],length:hi[1]-lo[1],angle:angle*180/Math.PI,top};
 }
 async locate(camera:string,prompt:any,viewPrompts:Record<string,any>={},regionIndex=0,supportZ=0.765,masks?:Record<string,string>) {
  this.supportZ=supportZ;
  const names=[camera,...Object.keys(this.robot.episode.read().images).filter(n=>n!==camera)],step=this.robot.episode.read().step;
  const {text,point,min_score:minScore}=prompt;
  const selectedRegion=async(name:string)=>{
   const {data,info}=await sharp(Buffer.from(masks![name],'base64')).toColourspace('srgb').ensureAlpha().raw().toBuffer({resolveWithObject:true});
   const view=this.camera(name);
   if(info.width!==view.width||info.height!==view.height)throw new Error('Supplied mask size differs from the original calibration');
   let area=0,rowSum=0,colSum=0,left=view.width,top=view.height,right=0,bottom=0;
   for(let i=0;i<info.width*info.height;i++) {
    const offset=i*info.channels,value=data[offset];
    if((value!==0&&value!==255)||data[offset+1]!==value||data[offset+2]!==value||data[offset+3]!==255)throw new Error('Supplied mask must contain opaque binary grayscale pixels');
    if(value){const row=Math.floor(i/view.width),col=i%view.width;area++;rowSum+=row;colSum+=col;left=Math.min(left,col);right=Math.max(right,col);top=Math.min(top,row);bottom=Math.max(bottom,row);}
   }
   if(!area)throw new Error('Supplied mask is empty');
   return {mask:masks![name],score:null,area,centroid:[rowSum/area,colSum/area],bbox:[left,top,right,bottom]};
  };
  const first=masks?{found:true,regions:[await selectedRegion(camera)]}:await this.segment(camera,prompt);
  const meta:any={camera,step,source:masks?'caller_mask_calibrated_silhouette_fit':'sam3_calibrated_silhouette_fit',unit:'m',support_z:this.supportZ,score:null,centroid:null,area_px:0,center_xyz:null,surface_xyz:null,geometry:null,fit:null,view_support:[],
   assumptions:{support_z:this.supportZ,height_search_cap_m:0.36,shape_candidates:['prism','hang'],robot_occlusion:'calibrated_link_boxes',mask_processing:masks?'caller_selected_pixels':'components_within_12_pixels_of_largest',footprint_completion:'fitted_circular_arcs',xy_domain:[[-0.72,-0.62],[0.72,0.52]]}};
  if(!first.found)return {...meta,found:false,note:'No mask in the requested camera'};
  const ordered=[...first.regions].sort((a,b)=>(b.score??0)-(a.score??0));
  if(regionIndex<0||regionIndex>=ordered.length)throw new Error('region_index is outside the returned anchor candidates');
  const selected=ordered[regionIndex];
  const decode=async(name:string,region:any,query:any)=>{
   const view=this.camera(name),{data,info}=await sharp(Buffer.from(region.mask,'base64')).removeAlpha().greyscale().raw().toBuffer({resolveWithObject:true});
   if(info.width!==view.width||info.height!==view.height)throw new Error('Mask size differs from calibration');
   return {camera:name,view,mask:masks?data:this.connectedMask(data,view.width),robotDepth:this.robotDepth(view),region,prompt:query} as Silhouette;
  };
  const anchor=await decode(camera,selected,masks?{source:'caller_supplied_mask'}:prompt);
  meta.score=selected.score;meta.centroid=selected.centroid.map((v:number)=>Math.floor(v));meta.area_px=anchor.mask.reduce((n,v)=>n+Number(v>0),0);
  const support=(s:Silhouette)=>({prompt:s.prompt,mask:s.region.mask,fit_mask_pixels:Array.from(s.mask.entries()).filter(([,value])=>value>0).map(([index])=>[Math.floor(index/s.view.width),index%s.view.width]),score:s.region.score,centroid:s.region.centroid,bbox:s.region.bbox,intrinsics:[[s.view.fx,0,s.view.cx],[0,s.view.fy,s.view.cy],[0,0,1]],camera_to_world:s.view.transform});
  const viewSupport:any[]=[{camera,candidates:ordered.length,selected:true,region_index:regionIndex,...support(anchor)}];meta.view_support=viewSupport;
  if(point&&meta.area_px>0.25*anchor.mask.length)return {...meta,found:true,views:1,note:'Large point-selected mask may describe the background or support; fitted geometry is unresolved'};
  const robotOverlap=this.overlap(anchor.mask,anchor.robotDepth),hidden=this.hiddenRim(anchor);
  if(robotOverlap>=0.5&&meta.score!==null&&meta.score<0.5)return {...meta,found:true,views:1,center_xyz:null,note:'Low-confidence mask overlaps the robot projection; fitted geometry is unresolved'};
  const table=this.supportZ,maxHeight=Math.min(0.36,anchor.view.transform[2][3]-table-0.05),shadow:V[]=[],pixels:V[]=[];
  meta.assumptions.height_search_limit_m=maxHeight;
  for(let i=0;i<anchor.mask.length;i++)if(anchor.mask[i]) {
   const row=Math.floor(i/anchor.view.width),col=i%anchor.view.width;
   const direction=rotate(anchor.view.transform,[(col-anchor.view.cx)/anchor.view.fx,-(row-anchor.view.cy)/anchor.view.fy,-1]);
   if(direction[2]<-1e-6){shadow.push(this.rayPlane(camera,row,col,table).xyz);pixels.push([row,col]);}
  }
  if(shadow.length<12||maxHeight<0.004)return {...meta,found:true,views:1,center_xyz:null,note:'Too little visible geometry above the support'};
  const cameraXY=position(anchor.view.transform),all=[...shadow,...shadow.map(p=>p.map((v,i)=>i<2?cameraXY[i]+(v-cameraXY[i])*(1-maxHeight/(cameraXY[2]-table)):v))];
  const xmin=Math.min(...all.map(p=>p[0]))-0.006,xmax=Math.max(...all.map(p=>p[0]))+0.006,ymin=Math.min(...all.map(p=>p[1]))-0.006,ymax=Math.max(...all.map(p=>p[1]))+0.006;
  if(xmax-xmin>1.6||ymax-ymin>1.6)return {...meta,found:true,views:1,center_xyz:null,note:'The projected mask spans more than the supported 1.6 m search width'};
  const spacing=Math.max(0.001,Math.sqrt((xmax-xmin)*(ymax-ymin)/40000)),nx=Math.ceil((xmax-xmin)/spacing)+1,ny=Math.ceil((ymax-ymin)/spacing)+1,total=nx*ny;
  const rise=new Float32Array(total),low=new Float32Array(total),high=new Float32Array(total);low.fill(NaN);high.fill(NaN);
  const levels=Math.floor(maxHeight/0.002),downwardRun=new Uint16Array((levels+1)*total);
  for(let i=0;i<total;i++) {
   const x=xmin+(i%nx)*spacing,y=ymin+Math.floor(i/nx)*spacing;
   let continuous=this.evidence(anchor,x,y,table)===1,run=0;
   for(let layer=0;layer<=levels;layer++) {
    const h=layer*0.002,allowed=this.evidence(anchor,x,y,table+h)!==0;
    run=allowed?run+1:0;downwardRun[layer*total+i]=run;
    if(allowed){if(!Number.isFinite(low[i]))low[i]=h;high[i]=h;}
    if(continuous&&allowed)rise[i]=h+0.002;else continuous=false;
   }
  }
  const base={x0:xmin,y0:ymin,spacing,nx,ny},cone:Solid={...base,low,high};
  const block=(height:number,kind:string):Solid=>{
   const bottom=new Float32Array(total),top=new Float32Array(total);bottom.fill(NaN);top.fill(NaN);
   const layer=Math.round(height/0.002);
   for(let i=0;i<total;i++) {
    if(kind==='prism'&&rise[i]>height){bottom[i]=0;top[i]=height;}
    if(kind==='hang'&&layer>=0&&layer<=levels){const length=downwardRun[layer*total+i];if(length>0){bottom[i]=(layer-length+1)*0.002;top[i]=layer*0.002;}}
   }
   return {...base,low:bottom,high:top};
  };
  const supports:Silhouette[]=[];
  for(const name of names.slice(1)) {
   if(masks) {
    if(masks[name]) {
     const candidate=await decode(name,await selectedRegion(name),{source:'caller_supplied_mask'});
     supports.push(candidate);
     viewSupport.push({camera:name,candidates:1,selected:true,selection:'caller',...support(candidate)});
    } else viewSupport.push({camera:name,candidates:0,selected:false,reason:'not_supplied'});
    continue;
   }
   const raw:any[]=[];
   const correction=viewPrompts[name];
   const query=correction??(text?{text,min_score:minScore}:null);
   if(query){const result=await this.segment(name,query);if(result.found)raw.push(...result.regions.map((region:any)=>({region,prompt:query})));}
   const destination=this.camera(name),depth=this.robotDepth(destination),tried:V[]=[];
   for(const h of correction?[]:[0.004,0.02,0.05,0.10,0.18]) {
    if(h>=maxHeight)continue;
    for(const pixel of [meta.centroid,pixels[Math.floor(pixels.length/4)],pixels[Math.floor(pixels.length*3/4)]]) {
     const p=this.rayPlane(camera,pixel[0],pixel[1],table+h).xyz,at=this.project(destination,p);
     if(!at||at[0]<0||at[1]<0||at[0]>=destination.height||at[1]>=destination.width||Number.isFinite(depth[at[0]*destination.width+at[1]])||tried.some(q=>Math.abs(q[0]-at[0])+Math.abs(q[1]-at[1])<12))continue;
     tried.push(at);const query={point:at,min_score:minScore},result=await this.segment(name,query);if(result.found)raw.push(...result.regions.map((region:any)=>({region,prompt:query})));
    }
   }
   const candidates:Silhouette[]=[];
   for(const {region,prompt:query} of raw) {
    const candidate=await decode(name,region,query),count=candidate.mask.reduce((n,v)=>n+Number(v>0),0);
    if(count<12||count>candidate.mask.length*0.3||(region.score!==null&&region.score<0.5&&this.overlap(candidate.mask,candidate.robotDepth)>0.85))continue;
    let seen=0,inside=0,k=0;const stride=Math.max(1,Math.floor(count/1500));
    for(let i=0;i<candidate.mask.length;i++)if(candidate.mask[i]&&k++%stride===0){seen++;if(this.cast(candidate.view,cone,Math.floor(i/candidate.view.width),i%candidate.view.width))inside++;}
    if(inside/Math.max(1,seen)>=0.8)candidates.push(candidate);
   }
   let winner:Silhouette|undefined,best=-1;
   for(const candidate of candidates)for(let height=0.004;height<=maxHeight;height+=0.008)for(const kind of ['prism','hang']){
    const fit=this.match(block(height,kind),candidate).fit;if(fit>best){best=fit;winner=candidate;}
   }
   if(winner&&best>=0.55)supports.push(winner);
   viewSupport.push({camera:name,candidates:raw.length,cone_candidates:candidates.length,best_fit:best<0?null:best,selected:!!winner&&best>=0.55,reason:winner&&best>=0.55?null:'cross_view_match_unresolved',...(winner&&best>=0.55?support(winner):{})});
  }
  const score=(height:number,kind:string)=>supports.reduce((sum,s)=>sum+this.match(block(height,kind),s).fit,0)/Math.max(1,supports.length);
  let kind='prism',height=0.004,fit:number|null=null;
  if(supports.length) {
   const best=new Map<string,{height:number;fit:number}>();
   for(const shape of ['prism','hang']) {
    let chosen={height:0.004,fit:-1};
    for(let h=0.004;h<=maxHeight;h+=0.004){const f=score(h,shape);if(f>chosen.fit)chosen={height:h,fit:f};}
    for(const h of [chosen.height-0.002,chosen.height+0.002])if(h>=0.002&&h<=maxHeight){const f=score(h,shape);if(f>chosen.fit)chosen={height:h,fit:f};}
    best.set(shape,chosen);
   }
   if(best.get('hang')!.fit>best.get('prism')!.fit+0.03)kind='hang';
   ({height,fit}=best.get(kind)!);
  }
  const known=masks?undefined:this.heights.get(this.memoryKey(text??''));let memoryStatus:string|null=null;
  if(known) {
   const old=block(known.height,known.shape),baseline=this.coverage(block(0.002,'prism'),anchor),oldCover=this.coverage(old,anchor);
   const bigger=known.height*1.25+0.01,tooTall=bigger<=maxHeight?this.coverage(block(bigger,known.shape),anchor):0;
   const compatible=known.height<=maxHeight&&oldCover>=baseline*0.9&&(known.shape==='hang'||tooTall<baseline*0.9);
   if(!compatible)memoryStatus='rejected';
   else if(!supports.length||fit!<known.fit-0.03){height=known.height;kind=known.shape;fit=known.fit;memoryStatus='used';}
  }
  if(!supports.length&&memoryStatus==='rejected'&&hidden)return {...meta,found:true,views:1,view_support:viewSupport,height_memory:memoryStatus,center_xyz:null,note:'The robot occludes the rim and the previous height no longer fits; current geometry is unresolved'};
  if(!supports.length&&memoryStatus!=='used') {
   const radial=shadow.map(p=>({angle:Math.atan2(p[0]-cameraXY[0],p[1]-cameraXY[1]),distance:Math.hypot(p[0]-cameraXY[0],p[1]-cameraXY[1])})).sort((a,b)=>a.angle-b.angle),limits:number[]=[];
   for(const fraction of [0.02,0.98]){
    const angle=radial[Math.floor((radial.length-1)*fraction)].angle,d=radial.filter(p=>Math.abs(p.angle-angle)<=1.5/anchor.view.fx).map(p=>p.distance).sort((a,b)=>a-b);
    if(d.length>=4){const near=d[Math.floor((d.length-1)*0.05)],far=d[Math.floor((d.length-1)*0.95)];if(far>near)limits.push((cameraXY[2]-table)*(far-near)/far);}
   }
   const limit=limits.length?limits.reduce((sum,v)=>sum+v,0)/limits.length+0.004:maxHeight,full=this.coverage(block(0.002,'prism'),anchor);
   for(let h=0.002;h<=Math.min(limit,maxHeight);h+=0.002){if(this.coverage(block(h,'prism'),anchor)<full*0.985)break;height=h;}
   kind='prism';fit=null;
  }
  const bodyHeight=height;
  let solid=block(height,kind);
  if(kind==='prism') {
   let ceiling=rise;
   if(supports.length&&memoryStatus!=='used') {
    const carved=new Float32Array(rise);
    for(let i=0;i<total;i++)if(rise[i]>0) {
     const x=xmin+(i%nx)*spacing,y=ymin+Math.floor(i/nx)*spacing;
     for(let z=0;z<rise[i];z+=0.002)if(supports.some(s=>this.evidence(s,x,y,table+z)===0)){carved[i]=z;break;}
    }
    let body=0,retained=0;for(let i=0;i<total;i++)if(rise[i]>height){body++;if(carved[i]>0)retained++;}
    if(retained>=body*0.85)ceiling=carved;
   }
   const columns=(h:number):Solid=>{const bottom=new Float32Array(total),top=new Float32Array(total);bottom.fill(NaN);top.fill(NaN);for(let i=0;i<total;i++)if(ceiling[i]>=0.004){bottom[i]=0;top[i]=Math.min(h,ceiling[i]-0.002);}return {...base,low:bottom,high:top};};
   if(supports.length&&memoryStatus!=='used') {
    let max=height;for(const h of ceiling)max=Math.max(max,h-0.002);
    const coverage=(h:number)=>supports.reduce((sum,s)=>sum+this.coverage(columns(h),s),0)/supports.length;
    const target=coverage(max);
    if(target-coverage(height)>=0.02){let a=height,b=max;while(b-a>0.002){const mid=Math.round((a+b)/0.004)*0.002;if(mid<=a||mid>=b)break;if(coverage(mid)>=target-0.01)b=mid;else a=mid;}height=b;}
   }
   solid=columns(height);
  }
  const measured:V[]=[];
  for(const pixel of pixels){const hit=this.cast(anchor.view,solid,pixel[0],pixel[1]);if(hit)measured.push(hit);}
  const rectangle=this.geometry(measured),wrap=(angle:number)=>((angle+90)%180+180)%180-90;
  if(!rectangle)return {...meta,found:true,views:1+supports.length,view_support:viewSupport,center_xyz:null,note:'Too few fitted surface samples above the support'};
  const top=rectangle.top-table;
  if(Math.abs(rectangle.x)>0.72||rectangle.y< -0.62||rectangle.y>0.52)return {...meta,found:true,views:1+supports.length,view_support:viewSupport,center_xyz:null,note:'Fitted geometry lies outside the stated XY domain'};
  const views=memoryStatus==='used'?1:1+supports.length,note=memoryStatus==='used'?'Using a compatible earlier cross-view height':!supports.length?`Single-view height guess from the tallest compatible upright prism; ${hidden?'the robot hides part of the boundary, so the footprint can be too small and off-centre; ':''}height is not constrained by another view`:fit!<0.9?'Partial or weak agreement between views; height can be centimetres off':null;
  const surface=point?this.cast(anchor.view,solid,point[0],point[1]):null;
  return {...meta,found:true,center_xyz:[rectangle.x,rectangle.y,table+top/2],top_z:table+top,height_m:top,length_m:Math.max(rectangle.width,rectangle.length),width_m:Math.min(rectangle.width,rectangle.length),long_axis_deg:wrap(rectangle.angle+(rectangle.width>=rectangle.length?0:90)),grasp_yaw:wrap(rectangle.angle+(rectangle.width<=rectangle.length?0:90)),views,view_support:viewSupport,height_memory:memoryStatus,fit,shape:kind,geometry:memoryStatus==='used'?'retained_height':supports.length?'multi_view_columns':'single_view_guess',note,grid_m:spacing,body_height_m:bodyHeight,surface_xyz:surface,_surface:{camera,step,mask:anchor.mask,width:anchor.view.width,solid,views,note,support_z:this.supportZ}};
 }
}
