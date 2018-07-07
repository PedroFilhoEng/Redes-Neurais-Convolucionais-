% % %  Teste AlexNet
       I = imread('C:\Users\PF\Pictures\teste.jpg');
              I = imresize(I,[227,227]);
              label = classify(rede,I);
              imshow(I)
           title(char(label))
 
% Teste GoogleNet
%  I = imread('C:\Users\PF\Pictures\teste.jpg');
%              I = imresize(I,[224,224]);
%              label = classify(net,I);
%              imshow(I)
%           title(char(label))




%       
% % 
% % %Teste feito com o grupo 
%          idx = randperm(numel(testImages.Files),4);
%          figure
%          for i = 1:4
%              subplot(2,2,i)
%              I = readimage(testImages,idx(i));
%              imshow(I)
%              label = predictedLabels(idx(i));
%              title(string(label));
%          end
% % %    

