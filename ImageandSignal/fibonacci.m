
%Here is a script that calculates the nth fibonacci number

n = 19;
sequencenumber = 0;
fibonaccinumber = 1;
if n > 2
    for i = 2:n
        oldfibonacci = fibonaccinumber;
        fibonaccinumber = fibonaccinumber + sequencenumber;
        sequencenumber = oldfibonacci;
    end
elseif n == 0
    fibonaccinumber = 0;
end
disp(fibonaccinumber)


%Here is a faster and better version for the fibonacci number
x = fibonaccinumbergenerator(n);
disp(x)

function x = fibonaccinumbergenerator(n)
    x = [0, 1] * [0, 1; 1, 1]^(n - 1) * [0; 1];
end