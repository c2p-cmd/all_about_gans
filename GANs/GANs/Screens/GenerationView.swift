//
//  GenerationView.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import SwiftUI

struct GenerationView: View {
    @Binding var vm: ViewModel
    let title: String
    
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    
    var body: some View {
        ScrollView {
            VStack {
                SliderView(vm: $vm)
                
                Divider()
                    .padding(.vertical, 10)
                
                if vm.isBusy {
                    BusyView()
                } else {
                    if vm.generatedImages.isEmpty {
                        ContentUnavailableView(
                            "Images not generated",
                            systemImage: "photo",
                            description: Text("Tap the button below to generate images using the selected model.")
                        )
                        .background(
                            Color.card,
                            in: RoundedRectangle(cornerRadius: 30, style: .continuous)
                        )
                    } else {
                        Text("Generated Images")
                            .font(.subheadline.bold())
                            .fontDesign(.rounded)
                        
                        LazyVGrid(columns: columns) {
                            ForEach(0..<vm.generatedImages.count, id: \.self) { idx in
                                let image: NativeImage = vm.generatedImages[idx]
                                ImageView(image: image)
                            }
                        }
                        .background(Color.background)
                    }
                }
            }
        }
        .background(Color.background)
        .scrollContentBackground(.automatic)
        .scrollBounceBehavior(.basedOnSize)
        .toolbar {
            #if os(macOS)
            let placement: ToolbarItemPlacement = .automatic
            #else
            let placement: ToolbarItemPlacement = .bottomBar
            #endif
            ToolbarItem(placement: placement) {
                Button("Generate Images", systemImage: "sparkles") {
#if targetEnvironment(simulator)
                    Task {
                        vm.isBusy = true
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
                        vm.generatedImages = []
                        for i in 0...15 {
                            vm.generatedImages.append(UIImage(named: "reconstructed_\(i)")!)
                        }
                        vm.isBusy = false
                    }
#else
                    vm.generateImages(count: Int(vm.imagesToGenerate))
#endif
                }
                .labelStyle(.titleAndIcon)
                .buttonStyle(.borderedProminent)
                .symbolEffect(.bounce.up.byLayer, options: .repeat(2))
            }
            
        }
        .alert(isPresented: $vm.showError, error: vm.error) { err in
            
        } message: { err in
            Text(err.message)
        }
        .navigationTitle(title)
        .task {
            vm.loadModel()
//            for i in 0...15 {
//                vm.generatedImages.append(UIImage(named: "reconstructed_\(i)")!)
//            }
        }
        .onDisappear {
            vm.modelInstance = nil
            vm.generatedImages = []
        }
    }
    
    var columns: [GridItem] {
        if horizontalSizeClass == .regular {
            [
                GridItem(.flexible(minimum: 75, maximum: 180)),
                GridItem(.flexible(minimum: 75, maximum: 180)),
                GridItem(.flexible(minimum: 75, maximum: 180)),
                GridItem(.flexible(minimum: 75, maximum: 180)),
            ]
        } else {
            [
                GridItem(.flexible(minimum: 75, maximum: 180)),
                GridItem(.flexible(minimum: 75, maximum: 180))
            ]
        }
    }
}

struct BusyView: View {
    var body: some View {
        VStack(alignment: .center, spacing: 20) {
            ProgressView()
            
            Text("Generating Images...")
                .font(.headline)
                .fontDesign(.rounded)
        }
        .padding(.all, 30)
        .background(
            Color.card,
            in: RoundedRectangle(cornerRadius: 30, style: .continuous)
        )
    }
}

struct SliderView: View {
    @Binding var vm: ViewModel
    
    var body: some View {
        VStack(alignment: .center) {
            Text("Images To Generate")
                .font(.subheadline)
                .fontWeight(.semibold)
                .fontDesign(.rounded)
            
            Slider(
                value: $vm.imagesToGenerate,
                in: 2...64,
                step: 1
            ) {
                Text("Images To Generate")
            } minimumValueLabel: {
                Image(systemName: "minus")
                    .onTapGesture {
                        withAnimation {
                            if vm.imagesToGenerate-1 >= 2 {
                                vm.imagesToGenerate -= 1
                            }
                        }
                    }
            } maximumValueLabel: {
                Image(systemName: "plus")
                    .onTapGesture {
                        withAnimation {
                            if vm.imagesToGenerate+1 <= 64 {
                                vm.imagesToGenerate += 1
                            }
                        }
                    }
            }
            .disabled(vm.isBusy)
            
            Text(Int(vm.imagesToGenerate), format: .number)
                .contentTransition(.numericText())
                .font(.subheadline)
                .fontDesign(.monospaced)
        }
        .padding(.horizontal)
    }
}

struct ImageView: View {
    let image: NativeImage
    
    var body: some View {
        imageView
            .scaledToFit()
            .cornerRadius(12)
            .padding(.bottom)
            .contextMenu {
                ShareLink(
                    item: imageView,
                    preview: SharePreview("Generated Image", icon: "bubbles.and.sparkles")
                )
            }
    }
    
    var imageView: Image {
#if os(macOS)
        Image(nsImage: image)
            .resizable()
#elseif os(iOS)
        Image(uiImage: image)
            .resizable()
#endif
    }
}

#Preview {
    @Previewable @State var vm = ViewModel()
    
    NavigationStack {
        GenerationView(
            vm: $vm,
            title: DomainModelTypes.mnistdcgan.description
        )
        .preferredColorScheme(.dark)
    }
}
