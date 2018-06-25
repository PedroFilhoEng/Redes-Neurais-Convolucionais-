%criando dataset de treinamento e valida��o.
imagens = imageDatastore('C:\Users\PF\Documents\treino','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.jpg','.png','.tiff','.bmp'});
%divide o dataset em 80% treino e 20% valida��o, a escolha das imagens dos
%grupos de treino e valida��o � aleat�ria.
[trainingImages, testImages] = splitEachLabel(imagens, 0.8, 'randomize');
%reiniciar a rede alexnet na vari�vel cnn.
cnn = alexnet; 
%a vari�vel layers, recebe as camadas rede cnn.
layers = cnn.Layers; 
%substitui a camadas 23 por outra camada totalmente conectada, com 2 labels (foco e sem foco). 
layers(23) = fullyConnectedLayer(2);
%substitui a camada 25 por outra camada de classifica��o.
layers(25) = classificationLayer;

%a vari�vel opts guarda os paramentros de treinamento da rede.
%sgdm = Descida gradiente estoc�stica com otimizador de impulso (SGDM).%Voc� pode especificar o valor de impulso, usando o argumento de par nome-valor 'Momentum' .
%initiallearnrate = Inicial de aprendizagem taxa utilizada para o treinamento, especificado como o par de separados por v�rgulas consistindo de 'InitialLearnRate' e um escalar positivo. O valor padr�o � 0,01 para o solver 'sgdm' e 0,001 para os 'rmsprop' e 'adam' solucionadores de problemas. Se a taxa de aprendizagem � muito baixa, ent�o o treinamento leva um longo tempo. Se a taxa de aprendizagem � muito alta, ent�o o treinamento pode chegar a um resultado de qualidade inferior ou divergem.
%maxepochs = n�mero m�ximo de �pocas a ser usadas para treinamento.
%minibatchsize = Tamanho do lote mini para usar para cada itera��o de forma��o, especificada como o par de separados por v�rgulas consistindo de 'MiniBatchSize' e um n�mero inteiro positivo. Um mini lote � um subconjunto do conjunto de treinamento que � usado para avaliar o gradiente da fun��o perda e atualizar os pesos. Ver estoc�stico Gradient Descent.
%verbose = Indicador para exibir informa��es de progresso de treinamento na janela de comando, especificado como o par de separados por v�rgulas consistindo de 'Verbose' e 1 (true) ou 0 (false).
%plots= 'training-progress'� tra�ar o progresso do treinamento. A trama mostra perda de mini lote e precis�o, perda de valida��o e precis�o e informa��es adicionais sobre o andamento do treinamento. O enredo tem um bot�o de stop  no canto superior direito. Clique no bot�o para parar de treinar e retornar o estado atual da rede. Para obter mais informa��es sobre a trama de progresso do treinamento, consulte Monitor progresso de forma��o de aprendizagem profunda.
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,...
    'MaxEpochs', 20, 'MiniBatchSize', 64,'Verbose',true,...
    'Plots','training-progress');

%redimensionando as imagens de entrada para 227 � 227 pixels, que � o que AlexNet espera. 
trainingImages.ReadFcn = @readFunctionTrain;

%trainamento da rede recebendo os par�metros e camadas
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
