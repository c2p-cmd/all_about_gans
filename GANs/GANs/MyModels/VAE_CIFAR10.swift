//
//  VAE_CIFAR10.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import Foundation
import MLX
import MLXNN

enum VAE_CIFAR10: DomainModel {
    static let latentDim = 200
    static let url = Bundle.main.url(forResource: "vae_10", withExtension: "safetensors")!
    
    static func loadPretrained() throws -> VAE {
        return try VAE.loadWeights(from: url)
    }
    
    class Encoder: Module, UnaryLayer {
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo var bn1: BatchNorm
        @ModuleInfo var pool1: MaxPool2d
        
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo var bn2: BatchNorm
        @ModuleInfo var pool2: MaxPool2d
        
        @ModuleInfo var conv3: Conv2d
        @ModuleInfo var bn3: BatchNorm
        @ModuleInfo var pool3: MaxPool2d
        
        @ModuleInfo var output_layer: Linear
        @ModuleInfo var bn_out: BatchNorm
        
        let flattenSize: Int = 4 * 4 * 256
        
        init(latentDim: Int = 200) {
            self.conv1 = Conv2d(inputChannels: 3, outputChannels: 64, kernelSize: 3, stride: 1, padding: 1)
            self.bn1 = BatchNorm(featureCount: 64)
            self.pool1 = MaxPool2d(kernelSize: 2, stride: 2)
            
            self.conv2 = Conv2d(inputChannels: 64, outputChannels: 128, kernelSize: 3, stride: 1, padding: 1)
            self.bn2 = BatchNorm(featureCount: 128)
            self.pool2 = MaxPool2d(kernelSize: 2, stride: 2)
            
            self.conv3 = Conv2d(inputChannels: 128, outputChannels: 256, kernelSize: 3, stride: 1, padding: 1)
            self.bn3 = BatchNorm(featureCount: 256)
            self.pool3 = MaxPool2d(kernelSize: 2, stride: 2)
            
            self.output_layer = Linear(flattenSize, latentDim)
            self.bn_out = BatchNorm(featureCount: latentDim)
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var o = silu(self.bn1(self.conv1(x)))
            o = self.pool1(o)
            
            o = silu(self.bn2(self.conv2(o)))
            o = self.pool2(o)
            
            o = silu(self.bn3(self.conv3(o)))
            o = self.pool3(o)
            
            o = o.reshaped([o.shape[0], -1])
            o = self.bn_out(self.output_layer(o))
            return silu(o)
        }
    }
    
    class Decoder: Module, UnaryLayer {
        @ModuleInfo var dense: Linear
        @ModuleInfo var bn_dense: BatchNorm
        
        @ModuleInfo var upconv1: UpsamplingConv2d
        @ModuleInfo var bn1: BatchNorm
        
        @ModuleInfo var upconv2: UpsamplingConv2d
        @ModuleInfo var bn2: BatchNorm
        
        @ModuleInfo var upconv3: UpsamplingConv2d
        @ModuleInfo var bn3: BatchNorm
        
        @ModuleInfo var final_conv: Conv2d
        
        init(latentDim: Int = 200) {
            self.dense = Linear(latentDim, 4 * 4 * 256)
            self.bn_dense = BatchNorm(featureCount: 4 * 4 * 256)
            
            self.upconv1 = UpsamplingConv2d(inputChannels: 256, outputChannels: 128, kernelSize: 3, padding: 1)
            self.bn1 = BatchNorm(featureCount: 128)
            
            self.upconv2 = UpsamplingConv2d(inputChannels: 128, outputChannels: 64, kernelSize: 3, padding: 1)
            self.bn2 = BatchNorm(featureCount: 64)
            
            self.upconv3 = UpsamplingConv2d(inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1)
            self.bn3 = BatchNorm(featureCount: 64)
            
            self.final_conv = Conv2d(inputChannels: 64, outputChannels: 3, kernelSize: 3, stride: 1, padding: 1)
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var o = silu(self.bn_dense(self.dense(x)))
            o = o.reshaped([-1, 4, 4, 256])
            o = silu(self.bn1(self.upconv1(o)))
            o = silu(self.bn2(self.upconv2(o)))
            o = silu(self.bn3(self.upconv3(o)))
            return sigmoid(self.final_conv(o))
        }
    }
    
    class VAE: Module {
        @ModuleInfo var encoder: Encoder
        @ModuleInfo var decoder: Decoder
        @ModuleInfo var proj_mu: Linear
        @ModuleInfo var proj_logvar: Linear
        
        init(latentDim: Int = 200) {
            self.encoder = Encoder(latentDim: latentDim)
            self.decoder = Decoder(latentDim: latentDim)
            self.proj_mu = Linear(latentDim, latentDim)
            self.proj_logvar = Linear(latentDim, latentDim)
        }
        
        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let features = self.encoder(x)
            let mu = self.proj_mu(features)
            let logvar = self.proj_logvar(features)
            let sigma = exp(0.5 * logvar)
            let epsilon = MLXRandom.normal(sigma.shape)
            let z = mu + sigma * epsilon
            let x_recon = self.decoder(z)
            return (x_recon, mu, logvar)
        }
        
        static func loadWeights(from url: URL) throws -> VAE {
            let params = try loadSafetensorsVAE(from: url)
            
            let vae = VAE(latentDim: latentDim)
//            print("VAE Parameters:")
//            print(vae.parameters())
            try vae.update(parameters: params, verify: .noUnusedKeys)
            
            eval(vae.parameters())
            
            return vae
        }
    }
}
