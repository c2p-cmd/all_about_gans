//
//  ContentView.swift
//  GANs
//
//  Created by Sharan Thakur on 13/11/25.
//

import MLX
import SwiftUI
#if canImport(AppKit)
typealias NativeImage = NSImage
#elseif canImport(UIKit)
typealias NativeImage = UIImage
#endif


struct ContentView: View {
    var body: some View {
        GANView()
            .navigationTitle("GANs with MLX")
    }
}

struct GANView: View {
    @State var generator: Generator?
    @State var batchSize: Int = 2
    @State var images: [NativeImage] = []
    
    var body: some View {
        List {
            Text("GAN View")
            Stepper("Batch Size: \(batchSize)", value: $batchSize, in: 1...64)
            Button(generator == nil ? "Load weights" : "Generate") {
                print("Button Pressed")
                guard let generator = generator else {
                    // Load weights
                    guard let fileURL = Bundle.main.url(forResource: "dcgan_generator_weights", withExtension: "safetensors") else {
                        print("Weights file not found")
                        return
                    }
                    do {
                        generator = try Generator.loadWeights(from: fileURL)
                    } catch {
                        print("Failed to load weights: \(error)")
                        print(String(describing: error))
                    }
                    return
                }
                let latentVectors = generateLatentVectors(batchSize: batchSize)
                let generatedImages = generator(latentVectors)
                self.images = generatedImages.compactMap { $0.toNativeImage() }
            }
            
            ForEach(0..<images.count, id: \.self) { i in
                let image = images[i]
                VStack {
                    Text("Image \(i + 1)")
                    getImage(nativeImage: image)
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: 200, maxHeight: 200)
                }
            }
        }
    }
    
    func getImage(nativeImage: NativeImage) -> some View {
#if os(macOS)
        Image(nsImage: nativeImage)
            .resizable()
#elseif os(iOS)
        Image(uiImage: nativeImage)
            .resizable()
#endif
    }
}

#Preview {
    ContentView()
}
