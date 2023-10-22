clear;
rng(13);

imAge(3,'cars.avi');
imAge(5,'cars.avi');
imAge(7,'cars.avi');
imAge(3,'flame.avi');
imAge(5,'flame.avi');
imAge(7,'flame.avi');

function imAge(F,file)
    cars = mmread(file,1:F);
    filename = string(file).split(".");
    filename = filename(1);
    H = 120;
    W = 240;
    gray_cars = zeros([H,W,F]);
    for i=1:F
        gray_cars(:,:,i) = double(rgb2gray(cars.frames(i).cdata(end-H+1:end,end-W+1:end,:)));
    end % Extract frames in grayscale

    E = zeros([H,W],'double');
    for i=1:F
       C(:,:,i) = double(randi([0,1],[H,W]));
       E = E + C(:,:,i).*gray_cars(:,:,i);
    end % get the coded snapshot
    
    E = E + randn([H,W])*2; % adding noise
    A=figure;   
    imshow(uint8(255*mat2gray(E)));

    t = title('Coded snapshot');
    t.FontSize = 10;
    saveas(A,['Q2_results/', char(filename), '/', num2str(F), '_coded_snapshot.jpg']);
    
    img = zeros([H,W,F],'double');
    count = zeros([H,W,F]);
    D = kron(dctmtx(8),dctmtx(8));
    psi = kron(eye(F),D); % get the block DCT matrix

    for i = 1:H-7
       for j = 1:W-7
           count(i:i+7,j:j+7,:) = count(i:i+7,j:j+7,:) + 1;
           A = [];
           for k = 1:F
               C_1 = reshape(C(i:i+7, j:j+7, k), 1, []);
               A = [A diag(C_1)];
           end   
           y = reshape(E(i:i+7,j:j+7), 1, []);
           d = double(OMP(y,A,psi));
           d = reshape(d, [8 8 F]);
           img(i:i+7, j: j+7,:) = img(i:i+7,j:j+7,:) + d;
       end 
    end
    
    img = img./count;
%     gray_cars = uint8(gray_cars);
    

    for i = 1:F
        Ab(i)=figure;   
        imshow(uint8([img(:,:,i); gray_cars(:,:,i)]));
        t = title(strcat("Reconstructed image of ", filename, " for ", num2str(i), " frame"));
        t.FontSize = 10;
        location = strcat('Q2_results/', filename , "/" ,num2str(F), "_", num2str(i),".png");
        saveas(Ab(i),location);
        squared_difference_norm = sum((gray_cars(:,:,i) - img(:,:,i)).^2, 'all');
        norm_frame = sum(gray_cars(:,:,i).^2, 'all');
        Ans = squared_difference_norm/norm_frame;
        fprintf('Relative MSE of frame %i: %i\n', i, Ans);
    end
    relative_MSE = (norm(gray_cars(:) - img(:))/norm(gray_cars(:)))^2;
    fprintf('Relative MSE of data: %i\n',relative_MSE);
    
    function img = OMP(y,A,psi)

        A1 = A*psi'; % To estimate DCT coeffecients
        [~,w] = size(A);
        r = single(y); theta = zeros([w,1],'double'); T = []; 
        while (norm(r)^2 > 9*4*64) % Many publications have stated norm(r) > 9*4*64 but both yield same results here
            pl = abs((r*A1)./(vecnorm(A1).^2));
            [~, max_column] = max(pl);
            T = [T max_column];
            theta(T) = pinv(A1(:, T)) * y'; % theta is column vector

            r = y - (A1(:,T)*theta(T))';
        end
        img = psi'*theta; % convert to canonical basis
    end
end