import {ReasoningEffortId} from '@deepseek-ai/dsh-llm';
import type {AgentRuntime,UpstreamBudget} from '../runtime/agent.ts';

export function agentRuntime(config:any,task:string,deadlineAt:number):AgentRuntime {
 return {
  policyPython:config.policyPython,sandboxConfig:config.sandboxConfig,ptcThreads:config.cpu.ptc_threads,
  model:config.model,reasoningEffort:ReasoningEffortId(config.reasoningEffort),maxTokens:config.maxTokens,maxRequests:config.maxRequests,maxCells:config.maxCells,maxNudges:config.maxNudges,
  episodeSeconds:config.episodeSeconds,cellSeconds:config.cellSeconds,ptcCpuSeconds:config.ptcCpuSeconds,apiRequestSeconds:config.apiRequestSeconds,
  jev:config.jev?{model:config.jev.model,endpoint:config.jev.endpoint,requestSeconds:config.jev.requestSeconds,apiKeyEnv:'JEV_API_KEY'}:null,
  demonstrations:config.demonstrations,
  taskIdentity:{task},deadlineAt,
  readUpstreamBudget:async():Promise<UpstreamBudget>=>({upstream_attempts_used:null,upstream_attempt_limit:null,upstream_attempts_recorded_at:null,upstream_attempts_note:'Provider-level usage is not supplied by the standalone adapter. These null values are unavailable measurements, not zero usage or an API budget.',jev_requests_used:null,jev_upstream_attempts_used:null})
 };
}
