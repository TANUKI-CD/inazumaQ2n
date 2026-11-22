% std2 of the Image Processing Toolbox, for an Octave without the image
% package: the standard deviation of all the elements of an array, with the
% divisor N - 1.
function s = std2(x)
  s = std(double(x(:)));
end
