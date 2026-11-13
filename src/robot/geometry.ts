import {mul,pose,quatPose,axisRotation,eye,type M,type V} from './math.ts';
import type {Kinematics,Side} from './kinematics.ts';

export type Body={link:string;fixed:boolean;transform:M;bounds:V[]};
export function bodies(kin:Kinematics,side:Side,joints:V,opening:number):Body[] {
 const base=kin.config.bases[side==='left'?0:1];
 const frames:Record<string,M>={base_link:quatPose([...base.default_root_pos,...base.default_root_rot])};
 for(let i=0;i<kin.chain.length;i++) {
  const joint=kin.chain[i];
  frames[joint.child]=mul(mul(frames[joint.parent],pose(joint.xyz,joint.rpy)),axisRotation(joint.axis,joints[i]));
 }
 const finger=kin.config.robot.gripper_scale[0]+opening*(kin.config.robot.gripper_scale[1]-kin.config.robot.gripper_scale[0]);
 for(const joint of kin.config.joints.filter((item:any)=>item.type==='fixed'||item.type==='prismatic')) {
  const offset=eye();
  if(joint.type==='prismatic')for(let i=0;i<3;i++)offset[i][3]=joint.axis[i]*finger;
  frames[joint.child]=mul(mul(frames[joint.parent],pose(joint.xyz,joint.rpy)),offset);
 }
 return Object.entries(kin.geometry.links).map(([link,value]:[string,any])=>({link,fixed:link==='base_link',transform:frames[link],bounds:value.bounds}));
}
