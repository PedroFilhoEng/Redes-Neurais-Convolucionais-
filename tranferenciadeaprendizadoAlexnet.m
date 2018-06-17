%O imageDatastore rotula automaticamente as imagens com base nos nomes das pastas e armazena os dados como um objeto ImageDatastore
images = imageDatastore('C:\Users\PF\Documents\dataset\treino','IncludeSubfolders',true,'FileExtensions',{'.jpg','.png'},'LabelSource','foldernames');

%Divida os dados em conjuntos de dados de treinamento e validação.Usa 70% das imagens para treinamento e 30% para validação. 
%O splitEachLabel divide o armazenamento de dados das imagens em dois novos armazenamentos de dados.
[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');


% Exibe algumas imagens de amostra.
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%inicia a alexnet na variável net
net = alexnet;

%mostra as camadas de rede da alexnet
net.Layers

% A primeira camada, a camada de entrada de imagem, requer imagens
%de entrada de tamanho 227 por 227 por 3, onde 3 é o número de canais de cor.
inputSize = net.Layers(1).InputSize

%salva todas as primeiras 22 camdas de rede da alexnet na variável
%layerTransfer(de 1 até a ultima-3)
layersTransfer = net.Layers(1:end-3);

%a variável numClasse guarda o n°de labels encontradas no grupo
%trainingImages(foco;sem foco), variável criada no datastore.
numClasses = numel(categories(trainingImages.Labels))


%Transferi as camadas para a nova tarefa de classificação, substituindo as 
%três últimas camadas por uma camada totalmente conectada, uma camada 
%softmax e uma camada de saída de classificação.  Define a camada totalmente 
%conectada para ter o mesmo tamanho que o número de classes nos novos dados. 
%Para aprender mais rapidamente nas novas camadas do que nas camadas 
%transferidas, basta aumentar os valores WeightLearnRateFactor e BiasLearnRateFactor 
%da camada totalmente conectada.
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


% %A rede requer imagens de entrada de tamanho 227 por 227 por 3, mas as
% %imagens nos datastores de imagem têm tamanhos diferentes. Use um
% datastore %de imagem aumentada para redimensionar automaticamente as
% imagens de %treinamento. Especifique operações de aumento adicionais a
% serem %executadas nas imagens de treinamento: inverta aleatoriamente as
% imagens %de treinamento ao longo do eixo vertical e as traduza
% aleatoriamente em %até 30 pixels na horizontal e na vertical. O aumento
% de dados ajuda a %impedir o overfitting da rede e a memorização dos
% detalhes exatos das %imagens de treinamento.


%pixelRange = [-30 30];
%imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);
%toda vez que tento usar essa função ou parametro augmentedImageDatastore, dá erro no maltlab('Undefined function or variable ')



%Para redimensionar automaticamente as imagens de validação sem executar o aumento de dados adicional,
%basta usar um armazenamento de dados de imagem aumentada sem especificar nenhuma operação adicional de pré-processamento.
%augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


%Especificando as opções de treinamento.
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

%treinamento
netTransfer = trainNetwork(trainingImages,layers,options);

%validação
predictedLabels = classify(netTransfer,validationImages);
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)
