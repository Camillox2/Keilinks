"""Mede compile/checkpoint na GPU real antes de escolher a configuração."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch
from treino.v4.config import get_model_config
from treino.v4.modelo import KeilinksV4


def run_case(model_name, context, steps, compile_enabled, checkpoint_mode):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA não disponível")
    config=get_model_config(model_name)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    model=KeilinksV4(config).cuda(); model.set_gradient_checkpointing(checkpoint_mode,2); model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True); executable=model
    if compile_enabled and hasattr(torch,"compile"):
        executable=torch.compile(model,mode="reduce-overhead",fullgraph=False)
    x=torch.randint(0,config.vocab_size,(1,context),device="cuda"); y=torch.randint(0,config.vocab_size,(1,context),device="cuda")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): _,loss=executable(x,y)
        loss.backward(); optimizer.step()
    torch.cuda.synchronize(); start=time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): _,loss=executable(x,y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    result={"model":model_name,"context":context,"steps":steps,"compile":compile_enabled,
            "checkpoint":checkpoint_mode,"seconds":elapsed,"tokens_per_second":steps*context/elapsed,
            "peak_vram_gb":torch.cuda.max_memory_allocated()/1e9,"last_loss":float(loss.item()),
            "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__}
    del executable,model,optimizer,x,y; torch.cuda.empty_cache(); gc.collect(); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="core_380m")
    p.add_argument("--context",type=int,default=1024); p.add_argument("--steps",type=int,default=10)
    p.add_argument("--output",default="checkpoints/v4/benchmark_rtx5050.json"); args=p.parse_args(); results=[]
    for compiled in (False,True):
        for mode in ("none","selective","full"):
            try:
                result=run_case(args.model,args.context,args.steps,compiled,mode); results.append(result); print(json.dumps(result,indent=2))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); results.append({"compile":compiled,"checkpoint":mode,"oom":True})
            except Exception as exc: results.append({"compile":compiled,"checkpoint":mode,"error":str(exc)})
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    valid=[r for r in results if "tokens_per_second" in r]
    if valid: print("Melhor:",json.dumps(max(valid,key=lambda r:r["tokens_per_second"]),indent=2))


if __name__=="__main__": main()
