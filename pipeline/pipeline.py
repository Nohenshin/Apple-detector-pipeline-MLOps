import kfp
from kfp import dsl

@dsl.pipeline(
    name='Apple Detection Pipeline',
    description='Train and evaluate Faster R-CNN on apple dataset'
)
def apple_pipeline(
    data_dir: str = '/app/data',
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 0.005
):
    train = dsl.ContainerOp(
        name='train',
        image='shigoruya.azurecr.io/apple-train:v2',  # thay bằng registry thật
        arguments=[
            '--data_dir', data_dir,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--lr', str(lr),
            '--output_dir', '/output'
        ],
        file_outputs={'model': '/output/best_model.pth'}
    )
    train.set_gpu_limit(1)   
    evaluate = dsl.ContainerOp(
        name='evaluate',
        image='shigoruya.azurecr.io/apple-evaluate:v2',
        arguments=[
            '--data_dir', data_dir,
            '--model_path', train.outputs['model'],
            '--batch_size', str(batch_size)
        ]
    ).after(train)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(apple_pipeline, 'apple_pipeline.yaml')