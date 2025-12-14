//
//  ViewModel.swift
//  GANs
//
//  Created by Sharan Thakur on 22/11/25.
//

import MLX
import MLXNN
import Observation
import SwiftUI

struct AppError: Error, LocalizedError, CustomStringConvertible {
    let title: String
    let message: String
    
    init(title: String, _ message: String) {
        self.title = title
        self.message = message
    }
    
    var description: String {
        "\(title): \(message)"
    }
    
    var errorDescription: String? { title }
}

enum DomainModelTypes: Int, CustomStringConvertible, CaseIterable {
    case mnistdcgan
    case cifar10vae
    case cifar10dcgan
    case fashionmnistvae
    case quickDraw8cvae
    
    var description: String {
        switch self {
        case .mnistdcgan:
            "MNIST DCGAN"
        case .cifar10vae:
            "CIFAR-10 VAE"
        case .cifar10dcgan:
            "CIFAR-10 DCGAN"
        case .quickDraw8cvae:
            "Quick Draw 8 CVAE"
        case .fashionmnistvae:
            "Fashion MNIST VAE"
        }
    }
    
    var infoText: LocalizedStringKey {
        switch self {
        case .mnistdcgan:
            "**LeCun, Y. et al. (1998)** Generates realistic handwritten digits (0-9) using a DCGAN architecture. This model excels at creating sharp 28x28 pixel images of numbers."
        case .cifar10vae:
            "**Krizhevsky, A. (2009)** Generates low-resolution (32x32) color images of common objects (like cars, birds, and frogs) using a VAE. It offers smooth transitions between different object classes."
        case .cifar10dcgan:
            "**Krizhevsky, A. (2009)** Generates 32x32 pixel color images from the CIFAR-10 dataset using a DCGAN. It often produces more visually distinct, though sometimes less abstract, images than the VAE."
        case .quickDraw8cvae:
            "**Google (2017)** Generates simple, black and white 28x28 sketches from 8 categories (like cat, basketball, mug) using a Conditional VAE (CVAE). You can select the class for generation. Here the model creates simple, cartoon-like sketches from Google's challenge."
        case .fashionmnistvae:
            "**Xiao, H. et al. (2017)** Generates 28x28 grayscale images of various clothing items (like shirts, bags, and boots) using a VAE. It produces smooth and varied images of apparel."
        }
    }
    
    func loadModel() throws -> Module {
        switch self {
        case .mnistdcgan:
            try DCGANMNIST.loadPretrained()
        case .cifar10vae:
            try VAE_CIFAR10.loadPretrained()
        case .cifar10dcgan:
            try DCGANCIFAR10.loadPretrained()
        case .quickDraw8cvae:
            try VAE_QuickDraw8.loadPretrained()
        case .fashionmnistvae:
            try VAE_Fashion_MNIST.loadPretrained()
        }
    }
    
    var latentDim: Int {
        switch self {
        case .mnistdcgan:
            DCGANMNIST.latentDim
        case .cifar10vae:
            VAE_CIFAR10.latentDim
        case .cifar10dcgan:
            DCGANCIFAR10.latentDim
        case .quickDraw8cvae:
            VAE_QuickDraw8.latentDim
        case .fashionmnistvae:
            VAE_Fashion_MNIST.latentDim
        }
    }
    
    @ViewBuilder
    var label: some View {
        switch self {
        case .mnistdcgan:
            Label("MNIST DCGAN", systemImage: "number.square.fill")
        case .cifar10vae:
            Label("CIFAR-10 VAE", systemImage: "photo.on.rectangle.angled")
        case .cifar10dcgan:
            Label("CIFAR-10 DCGAN", systemImage: "photo.on.rectangle")
        case .quickDraw8cvae:
            Label("Quick Draw 8 CVAE", systemImage: "pencil.and.outline")
        case .fashionmnistvae:
            Label("Fashion MNIST VAE", systemImage: "shoe.fill")
        }
    }
    
    var color: Color {
        switch self {
        case .mnistdcgan:
            Color.MNIST
        case .cifar10vae:
            Color.VAECIFAR_10
        case .cifar10dcgan:
            Color.DCGANCIFAR_10
        case .quickDraw8cvae:
            Color.vaeQuickDraw8
        case .fashionmnistvae:
            Color.vaeFashion
        }
    }
}

@Observable
final class ViewModel {
    var isBusy = false
    var showError = false
    var error: AppError?
    var currentModelType: DomainModelTypes?
    var modelInstance: Module?
    var generatedImages: [NativeImage] = []
    var imagesToGenerate: Float = 2
    
    func setError(_ error: AppError) {
        self.error = error
        self.showError = true
    }
    
    func loadModel() {
        Task.init {
            guard let currentModelType else {
                await MainActor.run {
                    setError(AppError(title: "No Model Selected", "Please select a model type before loading."))
                }
                return
            }
            
            await MainActor.run {
                self.isBusy = true
                self.error = nil
                self.showError = false
            }
            do {
                modelInstance = try currentModelType.loadModel()
            } catch {
                await MainActor.run {
                    setError(AppError(title: "Model Load Error", "Failed to load model: \(error.localizedDescription)"))
                }
                print(error)
            }
            await MainActor.run {
                self.isBusy = false
            }
        }
    }
    
    func generateImages(count imageCount: Int, label: VAE_QuickDraw8.Label? = nil) {
        Task.init {
            guard let modelInstance, let currentModelType else {
                await MainActor.run {
                    isBusy = false
                    setError(AppError(title: "Model not loaded", "Please load a model before generating images."))
                }
                return
            }
            
            await MainActor.run {
                self.isBusy = true
                self.error = nil
                self.showError = false
            }
            
            let latents = generateLatentVectors(batchSize: imageCount, latentDim: currentModelType.latentDim)
            
            if let mnist = modelInstance as? DCGANMNIST.Generator {
                let output = mnist(latents)
                await MainActor.run {
                    self.generatedImages = output.compactMap { $0.grayscaleToNativeImage(denormalize: denormalizeTanH(_:)) }
                }
            } else if let vae = modelInstance as? VAE_CIFAR10.VAE {
                let output = vae.decoder(latents)
                await MainActor.run {
                    self.generatedImages = output.compactMap { $0.rgbToNativeImage(denormalize: denormalizeSigmoid(_:)) }
                }
            } else if let cifargan = modelInstance as? DCGANCIFAR10.Generator {
                let output = cifargan(latents)
                await MainActor.run {
                    self.generatedImages = output.compactMap { $0.rgbToNativeImage(denormalize: denormalizeTanH(_:)) }
                }
            } else if let vaeFashion = modelInstance as? VAE_Fashion_MNIST.Decoder {
                let output = vaeFashion(latents)
                await MainActor.run {
                    self.generatedImages = output.compactMap { $0.grayscaleToNativeImage(denormalize: denormalizeSigmoid(_:)) }
                }
            } else if let vaeQuickDraw8 = modelInstance as? VAE_QuickDraw8.CVAE {
                let labels = MLXRandom.randInt(low: 0, high: VAE_QuickDraw8.Label.allCases.count, [imageCount])
                let output = vaeQuickDraw8.sample(num_samples: imageCount, labels: labels)
                await MainActor.run {
                    self.generatedImages = output.compactMap { $0.grayscaleToNativeImage(denormalize: denormalizeSigmoid(_:)) }
                }
            } else {
                print("Model instance is not a UnaryLayer nor VAE")
            }
            if self.generatedImages.isEmpty {
                await MainActor.run {
                    setError(AppError(title: "Image Generation Error", "Failed to convert generated images to NativeImage format."))
                }
            }
            
            await MainActor.run {
                self.isBusy = false
            }
        }
    }
}
