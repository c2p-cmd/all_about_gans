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
                if let currentModelType = vm.currentModelType {
                    VStack(spacing: 20) {
                        Text(currentModelType.description)
                            .font(.headline)
                            .fontDesign(.rounded)
                        
                        Text(currentModelType.infoText)
                            .font(.subheadline)
                            .fontDesign(.rounded)
                    }
                    .padding([.vertical, .horizontal], 20)
                    .background(
                        Color.card,
                        in: RoundedRectangle(cornerRadius: 36, style: .continuous)
                    )
                    .padding(.horizontal, 10)
                }
                
                Divider()
                    .padding(.vertical, 10)
                
                SliderView(vm: $vm)
                
                Divider()
                    .padding(.vertical, 10)
                
                if vm.isBusy {
                    BusyView()
                } else {
                    if vm.generatedImages.isEmpty {
                        Text("Tap the button below to generate images using the selected model.")
                            .foregroundStyle(.secondary)
                            .font(.subheadline)
                            .fontDesign(.rounded)
                            .padding(.horizontal)
                    } else {
                        Text("Generated Images")
                            .font(.headline)
                            .fontWeight(.regular)
                            .fontDesign(.rounded)
                            .foregroundStyle(.secondary)
                        
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
                generateButton
            }
        }
        .alert(isPresented: $vm.showError, error: vm.error) { err in
            
        } message: { err in
            Text(err.message)
        }
        .navigationTitle(title)
        .task {
#if targetEnvironment(simulator)
            print("Simulator")
#else
            vm.loadModel()
#endif
//            for i in 0...15 {
//                vm.generatedImages.append(UIImage(named: "reconstructed_\(i)")!)
//            }
        }
        .onDisappear {
            vm.modelInstance = nil
            vm.generatedImages = []
            vm.imagesToGenerate = 2
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
    
    var generateButton: some View {
        Button {
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
        } label: {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: "sparkles.2")
                
                Text("Generate Images")
            }
            .font(.subheadline)
            .padding(.horizontal, 10)
        }
        .labelStyle(.titleAndIcon)
        .buttonStyle(.borderedProminent)
        .buttonSizeIfAvailable(flexible: true)
        .symbolEffect(.bounce.up.byLayer, options: .repeat(2))
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
            Text("Select the number of images to generate")
                .font(.headline)
                .fontWeight(.semibold)
                .fontDesign(.rounded)
            
            Slider(
                value: $vm.imagesToGenerate,
                in: 2...64,
                step: 1
            ) {
                EmptyView()
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
    
    @State private var isPresented: Bool = false
    @State private var selection = PresentationDetent.large
    
    var body: some View {
        imageView
            .scaledToFit()
            .cornerRadius(12)
            .padding(.bottom)
            .onTapGesture {
                isPresented.toggle()
            }
            .sheet(isPresented: $isPresented) {
                NavigationStack {
                    VStack {
                        
                        Spacer()
                        
                        imageView
                            .scaledToFit()
                            .frame(height: 300)
                            .cornerRadius(12)
                            .padding(.bottom, 30)
                        
                        Spacer()
                        
                        HStack {
                            Spacer()
                            
                            ShareLink(
                                item: imageView,
                                preview: SharePreview("Generated Image", icon: "bubbles.and.sparkles")
                            ) {
                                Label("Share", systemImage: "square.and.arrow.up")
                            }
                            
                            Spacer()
                        }
                    }
                    .toolbar {
                        ToolbarItem(placement: .navigation) {
                            Button {
                                isPresented = false
                            } label: {
                                Label("Close", systemImage: "chevron.left")
                                    .labelStyle(.iconOnly)
                            }
                            .buttonStyle(.bordered)
                            .buttonBorderShape(.circle)
                        }
                    }
                }
                .presentationDetents([.large, .medium], selection: $selection)
                .presentationSizing(.automatic)
                .presentationDragIndicator(.visible)
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
        .onAppear {
            vm.currentModelType = .mnistdcgan
        }
    }
#if os(macOS)
    .frame(width: 1280 / 1.5, height: 720 / 1.5)
#endif
}
