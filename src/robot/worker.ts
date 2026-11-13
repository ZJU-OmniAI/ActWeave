import {parentPort,workerData} from 'node:worker_threads';
import {Kinematics} from './kinematics.ts';
import {Planner} from './planner.ts';
const kin=Object.assign(Object.create(Kinematics.prototype),{config:workerData.config,chain:workerData.chain});
const planner=new Planner(kin,workerData.initial);
const result=workerData.method==='trajectory'?planner.trajectory(...workerData.args as Parameters<Planner['trajectory']>):workerData.method==='plan'?planner.plan(...workerData.args as Parameters<Planner['plan']>):planner.prepareMove(...workerData.args as Parameters<Planner['prepareMove']>);
parentPort!.postMessage(result);
