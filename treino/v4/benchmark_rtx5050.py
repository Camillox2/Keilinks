"""Mede compile/checkpoint na GPU real antes de escolher a configuração."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch
from treino.v4.config import TrainConfig, get_model_config, get_train_config
from treino.v4.modelo import KeilinksV4
from treino.v4.treinar import build_optimizer


def run_case(model_name, profile, context, steps, compile_enabled, checkpoint_mode):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA não disponível")
    config=get_model_config(model_name)
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    train_config = get_train_config(profile)
    train_config = TrainConfig(
        **{**train_config.__dict__, "checkpoint_mode": checkpoint_mode}
    )
    model=KeilinksV4(config).cuda(); model.set_gradient_checkpointing(checkpoint_mode, train_config.checkpoint_every); model.train()
    optimizer=build_optimizer(model, train_config, torch.device("cuda")); executable=model
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
    p=argparse.ArgumentParser(); p.add_argument("--model",default="core_380m_modern")
    p.add_argument("--profile",default="rtx5050_380m")
    p.add_argument("--context",type=int,default=1024); p.add_argument("--steps",type=int,default=10)
    p.add_argument("--output",default="checkpoints/v4/benchmark_rtx5050.json")
    p.add_argument("--compile",action="store_true",help="Mede também o caminho torch.compile")
    p.add_argument("--checkpoint",choices=("none","selective","full"),default="selective")
    p.add_argument("--matrix",action="store_true",help="Testa todas as combinações; use apenas fora do treino")
    args=p.parse_args(); results=[]
    cases = (
        [(compiled, mode) for compiled in (False, True) for mode in ("none", "selective", "full")]
        if args.matrix else [(False, args.checkpoint)] + ([(True, args.checkpoint)] if args.compile else [])
    )
    for compiled, mode in cases:
        try:
            result=run_case(args.model,args.profile,args.context,args.steps,compiled,mode); results.append(result); print(json.dumps(result,indent=2))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); results.append({"compile":compiled,"checkpoint":mode,"oom":True})
        except Exception as exc: results.append({"compile":compiled,"checkpoint":mode,"error":str(exc)})
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    valid=[r for r in results if "tokens_per_second" in r]
    if valid: print("Melhor:",json.dumps(max(valid,key=lambda r:r["tokens_per_second"]),indent=2))


if __name__=="__main__": main()
