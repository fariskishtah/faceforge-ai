"""Local studio. Set FACEFORGE_CHECKPOINT to a trusted training checkpoint."""
import os,time,json
import pandas as pd
from pathlib import Path
import gradio as gr
from faceforge.inference import load,latent,render,interpolate
path=os.environ.get('FACEFORGE_CHECKPOINT','')
compare_path=os.environ.get('FACEFORGE_COMPARE_CHECKPOINT','')
compare_model=compare_cfg=None
if compare_path: compare_model,compare_cfg=load(compare_path)
model=cfg=None
if path:
    model,cfg=load(path)

def generate(seed,count):
    if model is None: raise gr.Error('No trained checkpoint configured. Set FACEFORGE_CHECKPOINT and restart.')
    start=time.perf_counter(); imgs=render(model,latent(int(seed),int(count),cfg['z_dim']))
    return imgs,f"Checkpoint: {os.path.basename(path)} · 64 × 64 · seed {int(seed)} · {time.perf_counter()-start:.2f} s"

def explore(a,b,steps):
    if model is None: raise gr.Error('No trained checkpoint configured. Set FACEFORGE_CHECKPOINT and restart.')
    za=latent(int(a),1,cfg['z_dim']); zb=latent(int(b),1,cfg['z_dim'])
    return render(model,interpolate(za,zb,int(steps)))

with gr.Blocks(title='FaceForge AI', theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'), css='body {background:#0b1220;} .gradio-container {max-width:1200px !important;}') as demo:
    gr.Markdown('# FaceForge AI\nSynthetic Face Generation & GAN Research Studio\n\n**AI-generated synthetic face imagery.** Generated images may resemble real individuals. Do not use for identity verification or impersonation.')
    if model is None: gr.Markdown('⚠️ No checkpoint configured. Generation is unavailable until a trusted trained model is supplied.')
    with gr.Tab('Generate'):
        with gr.Row():
            seed=gr.Number(label='Seed',value=42,precision=0); count=gr.Slider(1,32,4,step=1,label='Images')
        btn=gr.Button('Generate',variant='primary'); info=gr.Textbox(label='Run details'); gallery=gr.Gallery(label='Synthetic images',columns=4,allow_preview=True)
        btn.click(generate,[seed,count],[gallery,info])
    with gr.Tab('Latent explorer'):
        a=gr.Number(label='Start seed',value=42,precision=0); b=gr.Number(label='End seed',value=99,precision=0); steps=gr.Slider(2,32,8,step=1,label='Steps'); go=gr.Button('Interpolate'); result=gr.Gallery(columns=4); go.click(explore,[a,b,steps],result)
    with gr.Tab('Compare checkpoints'):
        gr.Markdown('The same latent vectors are used for a controlled visual comparison; independently trained models do not map them to matched identities.')
        compare_seed=gr.Number(label='Comparison seed',value=20261120,precision=0)
        compare_button=gr.Button('Compare')
        with gr.Row():
            old_gallery=gr.Gallery(label='Previous checkpoint',columns=4)
            new_gallery=gr.Gallery(label='Selected checkpoint',columns=4)
        def compare(seed):
            if model is None or compare_model is None: raise gr.Error('Set FACEFORGE_CHECKPOINT and FACEFORGE_COMPARE_CHECKPOINT.')
            if cfg['z_dim']!=compare_cfg['z_dim']: raise gr.Error('Latent dimensions differ; comparison unavailable.')
            z=latent(int(seed),8,cfg['z_dim'])
            return render(compare_model,z),render(model,z)
        compare_button.click(compare,[compare_seed],[old_gallery,new_gallery])
    with gr.Tab('Model quality'):
        if model is not None:
            nparams = sum(p.numel() for p in model.parameters())
            gr.Markdown(f'**Generator parameters:** {nparams:,}  |  **Resolution:** 64 × 64  |  **Latent dimensions:** {cfg["z_dim"]}\n\nFID: NOT EVALUATED · KID: NOT EVALUATED · Memorization: NOT EVALUATED')
            metrics_path = Path(path).parent / 'metrics.jsonl'
            if metrics_path.exists():
                rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
                frame = pd.DataFrame(rows).melt(id_vars=['step'], value_vars=['g_loss','d_loss'], var_name='objective', value_name='loss')
                gr.LinePlot(value=frame, x='step', y='loss', color='objective', title='Recorded training losses')
        else:
            gr.Markdown('No experiment loaded. FID, KID and memorization: NOT EVALUATED.')
    with gr.Tab('Training timeline'):
        gr.Markdown('Fixed-noise snapshots saved by the selected experiment. An early smoke run is not evidence of face quality.')
        snapshots = sorted(Path(path).parent.glob('fixed_epoch_*.png')) if path else []
        gr.Gallery(value=[str(x) for x in snapshots], label='Saved epoch grids', columns=3)
    with gr.Tab('Research notes'):
        gr.Markdown('The generator maps random vectors to images; the discriminator learns to distinguish training images from generated ones. Their losses alone do not measure image quality. See the local experiment artifacts for measured results. FID, KID and memorization are not evaluated unless documented in the experiment.')
if __name__=='__main__': demo.launch()
