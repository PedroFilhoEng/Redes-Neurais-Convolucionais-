  
%        I = imread('C:\Users\PF\Documents\dataset\teste\teste 3.jpg');
%        I = imresize(I,[227,227]);
%        label = classify(rede,I);
%        imshow(I)
%        title(char(label))

    idx = randperm(numel(testImages.Files),4);
    figure
    for i = 1:4
        subplot(2,2,i)
        I = readimage(testImages,idx(i));
        imshow(I)
        label = predictedLabels(idx(i));
        title(string(label));
    end
  

%  teste = imageDatastore('C:\Users\PF\Pictures');
% 
%   idx = randperm(numel(teste.Files),4);
%   figure
%   for i = 1:4
%       subplot(2,2,i)
%       I = readimage(teste,idx(i));
%       imshow(I)
%       label = predictedLabels(idx(i));
%       title(string(label));
%   end
%  
  