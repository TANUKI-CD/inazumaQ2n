% Run the q2n of a MATLAB pansharpening toolbox on one image pair.
%
% benchmarks/toolboxes.py calls this function under GNU Octave. The input
% file holds five int64 values, the height, the width, the number of bands,
% the block size and the block shift, and then the reference and the test
% image as doubles in column-major order. The output file receives three
% int64 values, the rows and the columns of the map and the number of
% rounds, and then, as doubles, the index, the map in column-major order and
% the seconds of every round. Only the call of q2n is timed.
%
% toolbox is the directory that holds q2n.m and its helpers. shims is added
% to the path only when the interpreter has no mean2 or std2, which belong
% to the Image Processing Toolbox of MATLAB and to the image package of
% Octave.
function run_q2n(input, output, toolbox, shims, rounds)
  addpath(toolbox);
  if ~exist('mean2') || ~exist('std2')
    addpath(shims);
  end
  fid = fopen(input, 'r');
  header = fread(fid, 5, 'int64')';
  count = prod(header(1:3));
  x = reshape(fread(fid, count, 'double'), header(1:3));
  y = reshape(fread(fid, count, 'double'), header(1:3));
  fclose(fid);
  seconds = zeros(1, rounds);
  for k = 1:rounds
    tic;
    [index, map] = q2n(x, y, header(4), header(5));
    seconds(k) = toc;
  end
  fid = fopen(output, 'w');
  fwrite(fid, [size(map, 1) size(map, 2) rounds], 'int64');
  fwrite(fid, index, 'double');
  fwrite(fid, map, 'double');
  fwrite(fid, seconds, 'double');
  fclose(fid);
end
