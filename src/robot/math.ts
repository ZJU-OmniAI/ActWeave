export type V = number[];
export type M = number[][];
export const eye = (): M => [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
export const mul = (a:M,b:M):M => a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));
export const add=(a:V,b:V):V=>a.map((v,i)=>v+b[i]);
export const sub=(a:V,b:V):V=>a.map((v,i)=>v-b[i]);
export const scale=(a:V,s:number):V=>a.map(v=>v*s);
export const dot=(a:V,b:V):number=>a.reduce((s,v,i)=>s+v*b[i],0);
export const norm=(a:V):number=>Math.sqrt(dot(a,a));
export const cross=(a:V,b:V):V=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function axisRotation(axis:V,t:number):M {
 const [x,y,z]=scale(axis,1/norm(axis)); const c=Math.cos(t),s=Math.sin(t),v=1-c;
 return [[x*x*v+c,x*y*v-z*s,x*z*v+y*s,0],[y*x*v+z*s,y*y*v+c,y*z*v-x*s,0],[z*x*v-y*s,z*y*v+x*s,z*z*v+c,0],[0,0,0,1]];
}
export function pose(xyz:V,rpy:V):M {
 const m=mul(mul(axisRotation([0,0,1],rpy[2]),axisRotation([0,1,0],rpy[1])),axisRotation([1,0,0],rpy[0]));
 xyz.forEach((v,i)=>m[i][3]=v); return m;
}
export function quatPose(p:V):M {
 const [w,x,y,z]=scale(p.slice(3),1/norm(p.slice(3)));
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),p[0]],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),p[1]],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),p[2]],[0,0,0,1]];
}
export const position=(m:M):V=>m.slice(0,3).map(row=>row[3]);
export const apply=(m:M,p:V):V=>m.slice(0,3).map(row=>dot(row.slice(0,3),p)+row[3]);
export const rotate=(m:M,p:V):V=>m.slice(0,3).map(row=>dot(row.slice(0,3),p));
export function inversePose(m:M):M { const r=eye();for(let i=0;i<3;i++){for(let j=0;j<3;j++)r[i][j]=m[j][i];r[i][3]=-dot(r[i].slice(0,3),position(m));}return r; }
export function orientation(pitch:number,yaw:number,roll:number):M { return pose([0,0,0],[roll*Math.PI/180,pitch*Math.PI/180,(90+yaw)*Math.PI/180]); }
export function angles(m:M):V {
 const pitch=Math.atan2(-m[2][0],Math.hypot(m[0][0],m[1][0]));
 const vertical=Math.hypot(m[0][0],m[1][0])<Math.sin(Math.PI/180);
 const heading=vertical?Math.atan2(-m[0][1],m[1][1]):Math.atan2(m[1][0],m[0][0]);
 const wrap=(degrees:number)=>((degrees+180)%360+360)%360-180;
 return [pitch*180/Math.PI,wrap(heading*180/Math.PI-90),vertical?0:wrap(Math.atan2(m[2][1],m[2][2])*180/Math.PI)];
}
export function rotationError(actual:M,target:M):V {
 const delta=mul(target,inversePose(actual));
 const trace=delta[0][0]+delta[1][1]+delta[2][2];
 let w:number;const v=[0,0,0];
 if(trace>0) {
  const s=2*Math.sqrt(1+trace);w=s/4;
  v[0]=(delta[2][1]-delta[1][2])/s;v[1]=(delta[0][2]-delta[2][0])/s;v[2]=(delta[1][0]-delta[0][1])/s;
 } else {
  const i=[0,1,2].reduce((best,k)=>delta[k][k]>delta[best][best]?k:best,0),j=(i+1)%3,k=(i+2)%3;
  const s=2*Math.sqrt(1+delta[i][i]-delta[j][j]-delta[k][k]);
  v[i]=s/4;v[j]=(delta[i][j]+delta[j][i])/s;v[k]=(delta[i][k]+delta[k][i])/s;w=(delta[k][j]-delta[j][k])/s;
 }
 const length=norm(v);
 return length<1e-12?scale(v,2):scale(v,(w<0?-1:1)*2*Math.atan2(length,Math.abs(w))/length);
}
