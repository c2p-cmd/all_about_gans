//
//  ViewModel.swift
//  GANs
//
//  Created by Sharan Thakur on 22/11/25.
//

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
    
    var description: String {
        switch self {
        case .mnistdcgan:
            "MNIST DCGAN"
        case .cifar10vae:
            "CIFAR-10 VAE"
        case .cifar10dcgan:
            "CIFAR-10 DCGAN"
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
            }
            await MainActor.run {
                self.isBusy = false
            }
        }
    }
    
    func generateImages(count imageCount: Int) {
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
