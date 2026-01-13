//
//  VAE_QuickDraw8.swift
//  GANs
//
//  Created by Sharan Thakur on 09/12/25.
//

import Foundation
import MLX
import MLXNN

enum VAE_QuickDraw8: DomainModel {
    static let latentDim = 32
    static let url = Bundle.main.url(forResource: "cvae_epoch_120", withExtension: "safetensors")!
    
    static func loadPretrained() throws -> CVAE {
        return try CVAE.loadWeights(from: url)
    }
    
    /*
     categories = [
         "cat",
         "basketball",
         "bird",
         "sword",
         "ice cream",
         "mug",
         "fish",
         "stop sign",
     ]
     */
    enum Label: Int, CaseIterable, CustomStringConvertible {
        case cat = 0
        case basketball = 1
        case bird = 2
        case sword = 3
        case ice_cream = 4
        case mug = 5
        case fish = 6
        case stop_sign = 7
        
        var description: String {
            switch self {
            case .cat:
                "Cat"
            case .basketball:
                "Baseball"
            case .bird:
                "Bird"
            case .sword:
                "Sword"
            case .ice_cream:
                "Ice Cream"
            case .mug:
                "Mug"
            case .fish:
                "Fish"
            case .stop_sign:
                "Stop Sign"
            }
        }
    }
    
    class Encoder: Module {
        let num_classes: Int
        
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo var conv3: Conv2d

        @ModuleInfo var cond_conv: Conv2d
        
        @ModuleInfo var fc_mu: Linear
        @ModuleInfo var fc_logvar: Linear
        
        init(num_classes: Int = 8, max_filters: Int = 64) {
            self.num_classes = num_classes
            
            self.conv1 = Conv2d(
                inputChannels: 1,
                outputChannels: Int(max_filters / 4),
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
            self.conv2 = Conv2d(
                inputChannels: Int(max_filters / 4),
                outputChannels: Int(max_filters / 2),
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
            self.conv3 = Conv2d(
                inputChannels: Int(max_filters / 2),
                outputChannels: max_filters,
                kernelSize: 3,
                stride: 1,
                padding: 1
            )
            
            self.cond_conv = Conv2d(
                inputChannels: max_filters,
                outputChannels: max_filters,
                kernelSize: 1
            )
            
            let flatten_dim = 7 * 7 * max_filters
            
            self.fc_mu = Linear(flatten_dim + num_classes, latentDim)
            self.fc_logvar = Linear(flatten_dim + num_classes, latentDim)
        }
        
        func callAsFunction(_ x: MLXArray, labels: MLXArray) -> (mu: MLXArray, logvar: MLXArray) {
            var h = relu(self.conv1(x))
            h = relu(self.conv2(x))
            h = relu(self.conv3(x))
            h = relu(self.cond_conv(x))
            
            h = reshaped(h, [h.shape[0], -1])
            
            let labels_one_hot = zeros([labels.shape[0], self.num_classes])
            labels_one_hot[arange(labels.shape[0]), labels] = MLXArray(1)
            
            h = concatenated([h, labels_one_hot], axis: 1)
            
            let mu = self.fc_mu(h)
            let logvar = self.fc_logvar(h)
            
            return (mu, logvar)
        }
    }
    
    class Decoder: Module {
        @ModuleInfo var fc: Linear
        
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo var conv3: Conv2d
        
        let num_classes: Int
        let max_filters: Int
        
        init(num_classes: Int = 8, max_filters: Int = 64) {
            self.num_classes = num_classes
            self.max_filters = max_filters
            
            self.fc = Linear(latentDim + num_classes, 7 * 7 * max_filters)
            
            self.conv1 = Conv2d(
                inputChannels: max_filters,
                outputChannels: Int(max_filters / 2),
                kernelSize: 3,
                padding: 1
            )
            self.conv2 = Conv2d(
                inputChannels: Int(max_filters / 2),
                outputChannels: Int(max_filters / 4),
                kernelSize: 3,
                padding: 1
            )
            self.conv3 = Conv2d(
                inputChannels: Int(max_filters / 4),
                outputChannels: 1,
                kernelSize: 3,
                padding: 1
            )
        }
        
        private func upsample(x: MLXArray, scale_factor: Int) -> MLXArray {
            var o = repeated(x, count: scale_factor, axis: 1)
            o = repeated(o, count: scale_factor, axis: 2)
            return o
        }
        
        func callAsFunction(_ x: MLXArray, labels: MLXArray) -> MLXArray {
            let labels_one_hot = zeros([labels.shape[0], self.num_classes])
            labels_one_hot[arange(labels.shape[0]), labels] = MLXArray(1)
            
            var z = concatenated([x, labels_one_hot], axis: 1)
            
            z = self.fc(z)
            z = reshaped(z, [z.shape[0], 7, 7, max_filters])
            
            z = relu(self.conv1(self.upsample(x: z, scale_factor: 2)))
            z = relu(self.conv2(self.upsample(x: z, scale_factor: 2)))
            z = sigmoid(self.conv3(z))
            
            return z
        }
    }
    
    class CVAE: Module {
        @ModuleInfo var encoder: Encoder
        @ModuleInfo var decoder: Decoder
        
        init(max_filters: Int = 64, num_classes: Int = 8) {
            self.encoder = Encoder(num_classes: num_classes, max_filters: max_filters)
            self.decoder = Decoder(num_classes: num_classes, max_filters: max_filters)
        }
        
        func reparametrize(x: (mu: MLXArray, logvar: MLXArray)) -> MLXArray {
            let (mu, logvar) = x
            let std = exp(0.5 * logvar)
            let eps = MLXRandom.normal(mu.shape)
            return mu + std * eps
        }
        
        func sample(num_samples: Int, labels: MLXArray) -> MLXArray {
            let z = MLXRandom.normal([num_samples, latentDim])
            return self.decoder(z, labels: labels)
        }
        
        func callAsFunction(_ x: MLXArray, labels: MLXArray) -> (x_recon: MLXArray, mu: MLXArray, logvar: MLXArray) {
            let encoder_out = self.encoder(x, labels: labels)
            let z = self.reparametrize(x: encoder_out)
            let x_recon = self.decoder(z, labels: labels)
            return (x_recon, encoder_out.mu, encoder_out.logvar)
        }
        
        static func loadWeights(from url: URL) throws -> CVAE {
            let params = try loadSafetensorsVAE(from: url)
            
            let cvae = CVAE(max_filters: 64, num_classes: Label.allCases.count)
            try cvae.update(parameters: params, verify: .noUnusedKeys)
            
            eval(cvae.parameters())
            
            return cvae
        }
    }
}
