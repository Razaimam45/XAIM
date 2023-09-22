import os
import torch
from torch import nn
from thop import profile, clever_format
import math
import numpy as np

# Lightweight neural network class to be used as student:
class Student(nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(64, 16, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(9 * 9 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def countParams(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: ", total_params)
    print("Trainable Params: ", trainable_params)

class Computation:

    @staticmethod
    def count_layers(model_path):
        # Count the total number of layers in the model
        model = torch.load(model_path).cuda()
        total_layers = sum(1 for _ in model.parameters())
        print("Total layers:", total_layers)
    
    @staticmethod
    def params_flops(model_path):
        input_size = (4, 9, 9)  # input size of the images
        input = torch.randn(1, *input_size)  # create a random tensor for input
        print(input.shape)

        model = torch.load(model_path).cuda().eval()
        # model = Student()
        # model.load_state_dict(model_x.state_dict())
        flops, params = profile(model.cuda(), inputs=(input.cuda(), ))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Model - Parameters: {params}, FLOPS: {flops}")

    @staticmethod
    def weight(model_path):
        model_size = os.path.getsize(model_path)

        # Convert size to human-readable format
        def convert_size(size_bytes):
            # 2^10 = 1024
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            size = round(size_bytes / p, 2)
            return f"{size} {size_name[i]}"

        # Print the size of the model
        print("Model Size:", convert_size(model_size))

    @staticmethod
    def inference_time(model_path):
        #[https://deci.ai/blog/measure-inference-time-deep-neural-networks/]
        model = torch.load(model_path)
        device = torch.device("cuda")
        model.to(device)
        dummy_input = torch.randn(1, 4, 9, 9, dtype=torch.float).to(device)

        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print('Inference Time:', mean_syn)

    @staticmethod
    def throughput(model_path):
        optimal_batch_size = 30
        model = torch.load(model_path)
        device = torch.device("cuda")
        model.to(device)
        dummy_input = torch.randn(optimal_batch_size, 4, 9, 9, dtype=torch.float).to(device)

        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
        Throughput = (repetitions * optimal_batch_size) / total_time
        print('Final Throughput:', Throughput)
    


if __name__ == "__main__":
    device = 'cuda'
    ccnn_model_path = '/l/users/raza.imam/ccnn_val_ar_dom.pth'
    arjunVit_model_path = '/l/users/raza.imam/ArjunViT_out_all4_again.pth'
    Vit_model_path = '/l/users/raza.imam/vit_12DH_VAD.pth'
    cnn2_model_path = '/l/users/raza.imam/cnn_2layers_var_ar_dom.pth'
    cnn1_model_path = '/l/users/raza.imam/cnn_1layers_var_ar_dom.pth'
    student_model_path = '/l/users/raza.imam/kd_ccnn_cnn2_VAD_Tpt25_lr3.pth'
    
    model_path = student_model_path
    
    teacher=torch.load(model_path)
    teacher=teacher.to(device)
    print(teacher)
    countParams(teacher)
    
    computation = Computation()
    computation.count_layers(model_path)
    computation.params_flops(model_path)
    computation.weight(model_path)
    computation.inference_time(model_path)
    computation.throughput(model_path)
