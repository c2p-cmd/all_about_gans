//
//  ContentView.swift
//  GANs
//
//  Created by Sharan Thakur on 13/11/25.
//

import SwiftUI

struct ContentView: View {
    @State private var vm = ViewModel()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                header
                
                Divider()
                    .padding(.vertical)
                
                buttons
            }
            .background(Color.background)
            .navigationDestination(item: $vm.currentModelType) {
                GenerationView(vm: $vm, title: $0.description)
            }
#if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
#endif
            .navigationTitle("ImageGen with MLX")
        }
    }
    
    @ViewBuilder
    var header: some View {
        VStack(spacing: 20) {
            Text("What are GANs?")
                .font(.headline)
                .fontDesign(.rounded)
            
            Text("Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed to generate new, synthetic data that closely resembles a given training dataset.")
                .font(.subheadline)
                .fontDesign(.rounded)
            
            Text("What are VAEs?")
                .font(.headline)
                .fontDesign(.rounded)
            
            Text("Variational Autoencoders (VAEs) are a type of generative model that learns to encode input data into a latent space and then decode it back to reconstruct the original data, allowing for the generation of new, similar data points.")
                .font(.subheadline)
                .fontDesign(.rounded)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 20)
        .background(
            Color.card,
            in: RoundedRectangle(cornerRadius: 36, style: .continuous)
        )
        .padding(.horizontal, 10)
    }
    
    @ViewBuilder
    var buttons: some View {
        VStack(alignment: .center, spacing: 30) {
            Text("Select a model to generate images:")
                .font(.headline)
                .fontDesign(.rounded)
                .padding(.bottom, 10)
            
            ForEach(DomainModelTypes.allCases, id: \.self) { modelType in
                Button {
                    self.vm.currentModelType = modelType
                } label: {
                    HStack(alignment: .center) {
                        modelType.label
                        Spacer()
                        Image(systemName: "chevron.right")
                    }
                    .padding(.horizontal, 5)
                    .padding(.vertical, 7.5)
                }
                .padding(.all, 5)
                .buttonStyle(.bordered)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 20)
        .background(
            Color.card,
            in: RoundedRectangle(cornerRadius: 36, style: .continuous)
        )
        .padding(.horizontal, 10)
    }
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}
