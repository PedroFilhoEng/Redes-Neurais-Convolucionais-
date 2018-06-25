%criando dataset de treinamento e validação.
imagens = imageDatastore('C:\Users\PF\Documents\treino','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.jpg','.png','.tiff','.bmp'});
%divide o dataset em 80% treino e 20% validação, a escolha das imagens dos
%grupos de treino e validação é aleatória.
[trainingImages, testImages] = splitEachLabel(imagens, 0.8, 'randomize');
%reiniciar a rede alexnet na variável cnn.
cnn = alexnet; 
%a variável layers, recebe as camadas rede cnn.
layers = cnn.Layers; 
%substitui a camadas 23 por outra camada totalmente conectada, com 2 labels (foco e sem foco). 
layers(23) = fullyConnectedLayer(2);
%substitui a camada 25 por outra camada de classificação.
layers(25) = classificationLayer;

%a variável opts guarda os paramentros de treinamento da rede.
%sgdm = Descida gradiente estocástica com otimizador de impulso (SGDM).%Você pode especificar o valor de impulso, usando o argumento de par nome-valor 'Momentum' .
%initiallearnrate = Inicial de aprendizagem taxa utilizada para o treinamento, especificado como o par de separados por vírgulas consistindo de 'InitialLearnRate' e um escalar positivo. O valor padrão é 0,01 para o solver 'sgdm' e 0,001 para os 'rmsprop' e 'adam' solucionadores de problemas. Se a taxa de aprendizagem é muito baixa, então o treinamento leva um longo tempo. Se a taxa de aprendizagem é muito alta, então o treinamento pode chegar a um resultado de qualidade inferior ou divergem.
%maxepochs = número máximo de épocas a ser usadas para treinamento.
%minibatchsize = Tamanho do lote mini para usar para cada iteração de formação, especificada como o par de separados por vírgulas consistindo de 'MiniBatchSize' e um número inteiro positivo. Um mini lote é um subconjunto do conjunto de treinamento que é usado para avaliar o gradiente da função perda e atualizar os pesos. Ver estocástico Gradient Descent.
%verbose = Indicador para exibir informações de progresso de treinamento na janela de comando, especificado como o par de separados por vírgulas consistindo de 'Verbose' e 1 (true) ou 0 (false).
%plots= 'training-progress'— traçar o progresso do treinamento. A trama mostra perda de mini lote e precisão, perda de validação e precisão e informações adicionais sobre o andamento do treinamento. O enredo tem um botão de stop  no canto superior direito. Clique no botão para parar de treinar e retornar o estado atual da rede. Para obter mais informações sobre a trama de progresso do treinamento, consulte Monitor progresso de formação de aprendizagem profunda.
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,...
    'MaxEpochs', 20, 'MiniBatchSize', 64,'Verbose',true,...
    'Plots','training-progress');

%redimensionando as imagens de entrada para 227 × 227 pixels, que é o que AlexNet espera. 
trainingImages.ReadFcn = @readFunctionTrain;

%trainamento da rede recebendo os parâmetros e camadas
rede = trainNetwork(trainingImages, layers, opts);


testImages.ReadFcn = @readFunctionTrain;
predictedLabels = classify(rede, testImages); 

accuracy = mean(predictedLabels == testImages.Labels);

%miniBatchSize = 10;
%numIterationsPerEpoch = floor(numel(trainingImagens.Labels)/miniBatchSize);
%options = trainingOptions('sgdm',...
%    'MiniBatchSize',miniBatchSize,...
%    'MaxEpochs',4,...
%    'InitialLearnRate',1e-4,...
%    'Verbose',false,...
%    'Plots','training-progress',...
%    'ValidationData',validationImages,...
%    'ValidationFrequency',numIterationsPerEpoch);

%treinamento
%netTransfer = trainNetwork(trainingImages,layers,options);
