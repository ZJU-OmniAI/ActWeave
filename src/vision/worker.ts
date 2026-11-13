import {parentPort,workerData} from 'node:worker_threads';
import {Episode} from '../bridge/episode.ts';
import {Kinematics} from '../robot/kinematics.ts';
import {Robot} from '../robot/control.ts';
import {Vision} from './vision.ts';
const episode=new Episode('');episode.observation=workerData.observation;
const kin=Object.assign(Object.create(Kinematics.prototype),{config:workerData.config,chain:workerData.chain,geometry:workerData.geometry});
const vision=new Vision(new Robot(episode,kin),workerData.url);
vision.heights=new Map(workerData.heights);vision.surfaces=workerData.surfaces;
const query=workerData.query;
let result:any;
if(workerData.method==='locate') {
 const value=await vision.locate(query.camera,query.prompt,query.view_prompts,query.region_index,query.support_z,query.masks);
 vision.remember(query.prompt.text??'',value);result=vision.sight(value,query.camera);
} else if(workerData.method==='back_project') {
 result=await vision.backProject(query.camera,query.row,query.col,query.support_z);
} else throw new Error('Unknown CAP operation');
parentPort!.postMessage({result,heights:Array.from(vision.heights),surfaces:vision.surfaces});
