% mean2 of the Image Processing Toolbox, for an Octave without the image
% package: the mean of all the elements of an array.
function m = mean2(x)
  m = mean(double(x(:)));
end
