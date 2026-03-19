import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelo.keilinks import Keilinks

def exportar():
    device = 'cpu'
    checkpoint_path = 'checkpoints/keilinks_flash.pt'
    
    if not os.path.exists(checkpoint_path):
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    modelo = Keilinks(checkpoint['config'])
    
    state = {k: v for k, v in checkpoint['modelo'].items() if 'embedding_posicao' not in k}
    modelo.load_state_dict(state, strict=False)
    modelo.eval()
    
    dummy_input = torch.randint(0, 32000, (1, 128), dtype=torch.long)
    caminho_saida = "checkpoints/keilinks.onnx"
    
    torch.onnx.export(
        modelo,
        dummy_input,
        caminho_saida,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )

if __name__ == '__main__':
    exportar()