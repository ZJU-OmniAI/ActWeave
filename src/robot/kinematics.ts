import { Matrix, solve } from 'ml-matrix';
import { readFileSync } from 'node:fs';
import { eye,mul,pose,quatPose,rotate,position,sub,norm,cross,rotationError, type V,type M } from './math.ts';
export type Side='left'|'right';
export class Kinematics {
 config:any;
 chain:any[];
 geometry:any;
 constructor(path:string) {
  this.config=JSON.parse(readFileSync(path,'utf8'));
  this.geometry=JSON.parse(readFileSync(new URL('../../configs/robot_geometry.json',import.meta.url),'utf8'));
  const ranges=JSON.parse(readFileSync(new URL('../../configs/robot_motion.json',import.meta.url),'utf8')).planning_joint_ranges_rad;
  this.chain=Array.from({length:6},(_,i)=>{
   const joint=this.config.joints.find((j:any)=>j.name===`joint${i+1}`);
   return {...joint,limits:[Math.max(joint.limits[0],ranges[i][0]),Math.min(joint.limits[1],ranges[i][1])]};
  });
 }
 frames(side:Side,q:V) {
  const base=this.config.bases[side==='left'?0:1];
  let transform=quatPose([...base.default_root_pos,...base.default_root_rot]);
  const axes:V[]=[],origins:V[]=[];
  for(let i=0;i<6;i++) {
   const joint=this.chain[i]; transform=mul(transform,pose(joint.xyz,joint.rpy));
   origins.push(position(transform));axes.push(rotate(transform,joint.axis));
   transform=mul(transform,axis(joint.axis,q[i]));
  }
  return {transform,axes,origins};
 }
 tcp(side:Side,q:V):M { const {transform}=this.frames(side,q);const offset=eye();offset[0][3]=this.config.robot.gripper_bias;return mul(transform,offset); }
 solve(side:Side,target:M,start:V,home:V=Array(6).fill(0)) {
  const seeds=[start,home,...[-Math.PI,Math.PI].map(turn=>start.map((q,i)=>i===5?q+turn:q)),this.chain.map(j=>(j.limits[0]+j.limits[1])/2)];
  const solveSeed=(seed:V)=>this.solveFrom(side,target,seed.map((q,i)=>Math.max(this.chain[i].limits[0],Math.min(this.chain[i].limits[1],q))));
  const results=seeds.map(solveSeed);
  if(!results.some(result=>result.reached)) {
   for(let joint=0;joint<5;joint++)for(const direction of [-1,1])results.push(solveSeed(start.map((q,i)=>q+(i===joint?direction*0.8:0))));
   for(let seed=1;seed<=6;seed++)results.push(solveSeed(start.map((q,i)=>q+0.6*Math.sin(seed*(i+1)*2.399963229728653))));
  }
  const reached=results.filter(result=>result.reached);
  const travel=(q:V)=>sub(q,start).reduce((sum,v,i)=>sum+Math.abs(v)*[2,2,1.5,1,1,0.5][i],0);
  return reached.length?reached.sort((a,b)=>travel(a.joints)-travel(b.joints))[0]:results.sort((a,b)=>(a.position_error_m+0.1*a.rotation_error_rad)-(b.position_error_m+0.1*b.rotation_error_rad))[0];
 }
 solveFrom(side:Side,target:M,start:V) {
  let q=[...start];
  for(let iteration=0;iteration<150;iteration++) {
   const f=this.frames(side,q),current=f.transform.map(row=>[...row]);
   for(let index=0;index<3;index++)current[index][3]+=f.transform[index][0]*this.config.robot.gripper_bias;
   const dp=sub(position(target),position(current)),dr=rotationError(current,target);
   if(norm(dp)<0.000001 && norm(dr)<0.00001)break;
   const columns=f.axes.map((a,i)=>[...cross(a,sub(position(current),f.origins[i])),...a.map(x=>x*0.2)]);
   const j=new Matrix(columns).transpose(),error=Matrix.columnVector([...dp,...dr.map(x=>x*0.2)]);
   const a=j.mmul(j.transpose());for(let i=0;i<6;i++)a.set(i,i,a.get(i,i)+0.0004);
   let step=j.transpose().mmul(solve(a,error)).to1DArray();const magnitude=norm(step);if(magnitude>0.25)step=step.map(x=>x*0.25/magnitude);
   q=q.map((v,i)=>Math.max(this.chain[i].limits[0],Math.min(this.chain[i].limits[1],v+step[i])));
  }
  const actual=this.tcp(side,q);const positionError=norm(sub(position(actual),position(target))),angleError=norm(rotationError(actual,target));
  return {joints:q,position_error_m:positionError,rotation_error_rad:angleError,reached:positionError<=0.002&&angleError<=Math.PI/180};
 }
 wristMount():M {
  const cam=this.config.robot.camera[0];let link=cam.link;const joints=[];
  while(link!==this.config.robot.ee_link){const joint=this.config.joints.find((j:any)=>j.child===link);if(!joint)throw new Error('Camera mount missing');joints.unshift(joint);link=joint.parent;}
  let t=eye();for(const joint of joints)t=mul(t,pose(joint.xyz,joint.rpy));
  const [rx,ry,rz]=cam.ori.map((v:number)=>v*Math.PI/180);
  const camera=mul(mul(axis([1,0,0],rx),axis([0,1,0],ry)),axis([0,0,1],rz));
  cam.pos.forEach((value:number,i:number)=>camera[i][3]=value);
  return mul(t,camera);
 }
}
import { axisRotation as axis } from './math.ts';
