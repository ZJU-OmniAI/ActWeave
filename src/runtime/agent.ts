import type {ReasoningEffortId} from '@deepseek-ai/dsh-llm';
import type {PythonLaunch} from '../../packages/ptc-runtime-python/index.ts';

export type UpstreamBudget = {upstream_attempts_used:number|null;upstream_attempt_limit:number|null;upstream_attempts_recorded_at:string|null;upstream_attempts_note:string|null;jev_requests_used:number|null;jev_upstream_attempts_used:number|null};
export type JevRuntime = {model:string;endpoint:string;requestSeconds:number;apiKeyEnv?:string};
export type AgentRuntime = {
 policyPython:string; sandboxConfig?:string; pythonLaunch?:PythonLaunch; ptcThreads:number;
 model:string; reasoningEffort:ReasoningEffortId; maxTokens:number; maxRequests:number|null; maxCells:number; maxNudges:number;
 episodeSeconds:number; cellSeconds:number; ptcCpuSeconds:number; apiRequestSeconds:number;
 jev:JevRuntime|null;
 demonstrations?:{enabled:boolean;directory?:string};
 taskIdentity:{task:string};
 deadlineAt:number; readUpstreamBudget:()=>Promise<UpstreamBudget>;
};
