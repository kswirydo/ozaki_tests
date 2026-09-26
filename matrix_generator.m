
function matrix_generator
m =  1024;
n =  1024;
log10KA = 1;
for i = 1:16
    log10KA = 1+i; % 10
    %
    U = randn(m,n); [U,~]=qr(U,0);
    V = randn(n,n); [V,~]=qr(V,0);
    S = diag( 10.^( linspace( 0, log10KA, n ) ) );
  %  keyboard
    A = U * S * V';

    absA = abs(A);
    expA = floor(log2(absA));
    expA(absA == 0) = NaN;
    Rglobal = max(expA, [], "all", "omitnan") ...
        - min(expA, [], "all", "omitnan");

    Rrow = max(expA, [], 2, "omitnan") ...
        - min(expA, [], 2, "omitnan");

    Rcol = max(expA, [], 1, "omitnan") ...
        - min(expA, [], 1, "omitnan");

    fprintf("Requested condition number: 1e%d\n", log10KA);
    fprintf("Theoretical R(S): %d bits\n", ...
        floor(log10KA * log2(10)));
    fprintf("Global R(A): %d bits\n", Rglobal);
    fprintf("Maximum row range: %d bits\n", max(Rrow));
    fprintf("Maximum column range: %d bits\n", max(Rcol));
    fprintf("Median row range: %.1f bits\n", median(Rrow));
    fprintf("Median column range: %.1f bits\n", median(Rcol));
    %   filename = sprintf('M_cond_1e%d.txt',log10KA);
    %   writematrix( A, filename);

    clear U S V;
end
end

function write_matrix_to_file(filename, A)
[n,m] = size(A);
fileID = fopen(filename, 'w');
fprintf(fileID, '%d %d \n', n, m);
for i = 1:n
    for j = 1:m
        fprintf(fileID, ' %16.16f ', A(i, j));

    end
    fprintf(fileID, '\n'); % Newline at the end of each row
end
fclose(fileID);

end