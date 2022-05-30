function status = decaes(varargin)
%DECAES (DE)composition and (C)omponent (A)nalysis of (E)xponential (S)ignals
% Call out to the DECAES command line tool. The DECAES.jl Julia package will
% be installed automatically, if necessary.
% 
% See the online documentation for more information on DECAES:
%   https://jondeuce.github.io/DECAES.jl/dev/
% 
% See the mwiexamples github repository for myelin water imaging examples,
% including sample data:
%   https://github.com/jondeuce/mwiexamples
% 
% If you use DECAES in your research, please cite our work:
%   https://doi.org/10.1016/j.zemedi.2020.04.001
% 
% INPUTS:
%   Input argument syntax mimics that of the DECAES command line interface.
%   Arguments must be strings, numeric values, or arrays of numeric values.
%   For arrays, each element is treated as a separate input argument.
%   All arguments will be forwarded to DECAES, with the exception of the
%   following Matlab-specific flags:
% 
%   --runtime:  Path to Julia binary runtime. Defaults to 'julia'.
%   --threads:  Number of computational threads for DECAES to use.
%               Defaults to 'auto', deferring to Julia to choose the optimal value.
%   --project:  Julia project into which DECAES is installed. By default, a
%               folder '.decaes' is created in the directory containing this file.
%   --server:   (Experimental) start a local Julia server for running DECAES.
%               The first time DECAES is called with the --server flag, a background
%               Julia process will be instantiated with DECAES loaded. Subsequent
%               calls to DECAES with the --server flag will run DECAES on this
%               local Julia server, leading to greatly reduced startup time and
%               much friendlier interactive use. The Julia server will be killed
%               automatically when Matlab exits. Do not use this flag if DECAES
%               only needs to be run once.
% 
% OUTPUTS:
%   status:     (optional) System call status; see SYSTEM for details
% 
% EXAMPLES:
%   Run DECAES multithreaded on 'image.nii.gz' using command syntax:
%     * We specify an output folder named 'results' using the '--output'
%       flag; this folder will be created if it does not exist
%     * We pass a binary mask file 'image_mask.mat' using the --mask flag;
%       note that the mask file type need not match the image file type
%     * We specify that both T2 distribution calculation and T2 parts
%       analysis should be performed with the --T2map and --T2part flags
%     * The required arguments echo time, number of T2 bins, T2 Range,
%       small peak window, and middle peak window are set using the --TE,
%       --nT2, --T2Range, --SPWin, and --MPWin flags, respectively
%     * We specify the L-curve method for choosing the regularization
%       parameter using the --Reg flag
%     * Lastly, we indicate that the regularization parameters should be
%       saved using the --SaveRegParam flag
% 
%       decaes image.nii.gz --output results --mask image_mask.mat --T2map --T2part --TE 7e-3 --nT2 60 --T2Range 10e-3 2.0 --SPWin 10e-3 25e-3 --MPWin 25e-3 200.0e-3 --Reg lcurve --SaveRegParam
% 
%   Run the same command using function syntax:
% 
%       decaes('image.nii.gz', '--output', 'results', '--mask', 'image_mask.mat', '--T2map', '--T2part', '--TE', 7e-3, '--nT2', 60, '--T2Range', [10e-3, 2.0], '--SPWin', [10e-3, 25e-3], '--MPWin', [25e-3, 200.0e-3], '--Reg', 'lcurve', '--SaveRegParam')
% 
%   Create a settings file called 'settings.txt' containing the settings
%   from the above example (note: only one value or flag per line):
% 
%       image.nii.gz
%       --output
%       results
%       --mask
%       image_mask.mat
%       --T2map
%       --T2part
%       --TE
%       7e-3
%       --nT2
%       60
%       --T2Range
%       10e-3
%       2.0
%       --SPWin
%       10e-3
%       25e-3
%       --MPWin
%       25e-3
%       200.0e-3
%       --Reg
%       lcurve
%       --SaveRegParam
% 
%   Run the example using the above settings file 'settings.txt' on a local Julia server:
% 
%       decaes --server @settings.txt
% 
%   Note the separation of the Matlab-specific flags from the DECAES settings file.
% 
% This version of decaes.m was written for DECAES v0.4.5.
% DECAES was written by Jonathan Doucette (jdoucette@physics.ubc.ca).
% Original MATLAB implementation is by Thomas Prasloski (tprasloski@gmail.com).

    [opts, decaes_args] = parse_args(varargin{:});

    % Instantiate the default project, if necessary
    jl_instantiate_project(opts);

    % Call DECAES command line interface
    if opts.server
        st = jl_call_decaes_server(opts, decaes_args);
    else
        st = jl_call_decaes(opts, decaes_args);
    end

    % Return system call status, if requested
    if nargout > 0
        status = st;
    end

end

function jl_instantiate_project(opts)

    if strcmpi(opts.project, default_project)
        % Check if project folder exists with a Project.toml file
        proj_folder_exists = exist(opts.project, 'dir');
        proj_file_exists = exist(fullfile(opts.project, 'Project.toml'), 'file');

        if proj_folder_exists && proj_file_exists
            return
        end
        if ~proj_folder_exists
            mkdir(opts.project); % make new default project folder
        end
        if ~proj_file_exists
            fclose(fopen(fullfile(opts.project, 'Project.toml'), 'w')); % make Project.toml file
        end

        % Install DECAES into project
        install_script = jl_make_script('DECAES', {}, true);
        cleanup_install_script = onCleanup(@() delete([install_script, '*']));
        cmd = [jl_build_cmd(opts), ' ', install_script];
        system(cmd, '-echo');
    end

end

function cmd = jl_build_cmd(opts)

    % Set Julia binary path and flags
    jl_cmd_args = {
        opts.runtime
        '--startup-file=no'
        '--quiet'
        '--optimize=3'
        sprintf('--threads=%s', opts.threads)
    };
    if ~isempty(opts.project)
        jl_cmd_args{end+1} = sprintf('--project=%s', opts.project); %#ok
    end
    cmd = join(jl_cmd_args, ' ');
    cmd = cmd{1};

end

function st = jl_call_decaes(opts, decaes_args)

    % Create temporary script for calling DECAES entrypoint function
    decaes_script = jl_make_script('DECAES', 'main()');
    cleanup_decaes_script = onCleanup(@() delete([decaes_script, '*']));

    % Create system command, forwarding decaes_args to julia
    cmd = [jl_build_cmd(opts), ' ', decaes_script, ' ', decaes_args];

    % Call out to julia
    [st, ~] = system(cmd, '-echo');

end

function st = jl_call_decaes_server(opts, decaes_args)

    % Cleanup function for sending kill signal to DECAES server
    function kill_server()
        fprintf('[ Info: Killing DECAES server\n');
        kill_script = jl_make_script('DaemonMode', 'sendExitCode()');
        cleanup_kill_script = onCleanup(@() delete([kill_script, '*']));
        cmd = [jl_build_cmd(opts), ' ', kill_script];
        system(cmd, '-echo');
    end

    mlock % Prevent Matlab from clearing persistent variables via e.g. `clear all`
    persistent cleanup_server % DECAES server cleanup object

    if isempty(cleanup_server)
        % Create and run system command to start DECAES server
        fprintf('[ Info: Starting DECAES server\n');
        server_script = jl_make_script('DaemonMode', 'serve()');
        cleanup_server_script = onCleanup(@() delete([server_script, '*']));
        cmd = [jl_build_cmd(opts), ' ', server_script, ' &'];
        system(cmd, '-echo');

        % Wait for server ping
        while ~jl_server_ping(opts)
            pause(1)
        end

        % Kill server on Matlab exit
        cleanup_server = onCleanup(@() kill_server);
    end

    % Call DECAES entrypoint function on DECAES server
    daemon_script = jl_make_script('DaemonMode', 'runargs()');
    decaes_script = jl_make_script('DECAES', 'main()');
    cleanup_daemon_script = onCleanup(@() delete([daemon_script, '*']));
    cleanup_decaes_script = onCleanup(@() delete([decaes_script, '*']));
    cmd = [jl_build_cmd(opts), ' ', daemon_script, ' ', decaes_script, ' ', decaes_args];
    [st, ~] = system(cmd, '-echo');

end

function succ = jl_server_ping(opts)

    % Create temporary file and try to delete file from julia server
    daemon_script = jl_make_script('DaemonMode', {
        'redirect_stdout(devnull) do'
        '    redirect_stderr(devnull) do'
        '        runargs()'
        '    end'
        'end'
    });
    ping_script = jl_make_script({}, 'rm(ARGS[1]; force = true)');
    ping_file = tempname;
    cleanup_daemon_script = onCleanup(@() delete([daemon_script, '*']));
    cleanup_ping_script = onCleanup(@() delete([ping_script, '*']));
    cleanup_ping_file = onCleanup(@() delete([ping_file, '*']));
    cmd = [jl_build_cmd(opts), ' ', daemon_script, ' ', ping_script, ' ', ping_file];
    system(cmd, '-echo');
    succ = ~exist(ping_file, 'file');

end

function jl_script = jl_make_script(pkgs, body, install)

    if nargin < 3; install = false; end
    if nargin < 2; body = {}; end
    if nargin < 1; pkgs = {}; end

    if ischar(pkgs); pkgs = {pkgs}; end
    if ischar(body); body = {body}; end

    % Create temporary helper Julia script
    jl_script = [tempname, '.jl'];
    fid = fopen(jl_script, 'w');
    cleanup_fid = onCleanup(@() fclose(fid));

    for ii = 1:length(pkgs)
        fprintf(fid, jl_using_package_str(pkgs{ii}, install));
    end
    for ii = 1:length(body)
        fprintf(fid, [body{ii}, '\n']);
    end

end

function jl_str = jl_using_package_str(pkg, install)

    if nargin < 2; install = false; end

    jl_str = {
        'import Pkg'
        'if __INSTALL__ != 0'
        '    println("[ Info: Installing __PACKAGE__")'
        '    Pkg.add("__PACKAGE__"; io = devnull)'
        '    @eval using __PACKAGE__'
        'end'
        'try'
        '    @eval using __PACKAGE__'
        'catch e'
        '    println("[ Info: Installing __PACKAGE__")'
        '    Pkg.add("__PACKAGE__"; io = devnull)'
        '    @eval using __PACKAGE__'
        'end'
    };
    jl_str = sprintf('%s\n', jl_str{:});
    jl_str = strrep(jl_str, '__PACKAGE__', pkg);
    jl_str = strrep(jl_str, '__INSTALL__', num2str(install));

end

function x = check_positive_int(x)

    % Allow char inputs
    if ischar(x)
        x = str2double(x);
    end

    % Return NaN if x is not a positive integer
    if ~(isnumeric(x) && isscalar(x) && ~isnan(x) && x > 0 && x == round(x))
        x = NaN;
    end

end

function x = threads_string(x)

    % Allow char inputs
    if ischar(x) && strcmpi(x, 'auto')
        x = 'auto';
    else
        x = check_positive_int(x);
        if ~isnan(x)
            x = num2str(x);
        end
    end

end

function proj = default_project()

    [decaes_dir, ~, ~] = fileparts(mfilename('fullpath'));
    proj = fullfile(decaes_dir, '.decaes');

end

function runtime = try_find_julia_runtime()

    % Cached julia runtime location
    persistent JULIA_RUNTIME

    if ~isempty(JULIA_RUNTIME)
        runtime = JULIA_RUNTIME;
        return
    end

    % Try to find the default system julia
    if isunix
        [st, res] = system('which julia');
    elseif ispc
        [st, res] = system('where julia');
    end

    if st == 0
        runtime = strtrim(res);
    else
        runtime = 'julia'; % default value
    end

    % Try calling Julia
    jl_sys_bindir = tempname;
    cleanup_fid = onCleanup(@() delete([jl_sys_bindir, '*']));

    cmd = sprintf('%s --startup-file=no --quiet -e ''open(raw"%s"; write = true) do io; print(io, joinpath(Base.Sys.BINDIR, Base.julia_exename())); end''', runtime, jl_sys_bindir);
    [st, ~] = system(cmd);

    if st == 0
        runtime = strtrim(fileread(jl_sys_bindir));
    end

    % Cache julia runtime path
    JULIA_RUNTIME = runtime;

end

function [opts, decaes_args] = parse_args(varargin)

    % Check for deprecated syntax for setting number of Julia threads
    if ~isempty(varargin)
        nthreads = check_positive_int(varargin{1});
        if ~isnan(nthreads)
            varargin = {varargin{2:end}, '--threads', nthreads};
            warning(['Passing the number of threads as first argument is deprecated syntax.\n' ...
                     'Use flag/value pairs instead: ''decaes --threads %d ...'''], nthreads);
        end
    end

    % Verify number of input arguments
    if numel(varargin) < 1
        error('Must specify input image or settings file');
    end

    % Split varargin into Matlab and DECAES arguments
    mat_args = {};
    decaes_args = '';
    ii = 1;
    while ii <= numel(varargin)
        arg = varargin{ii};
        if ischar(arg)
            switch lower(arg)
                case '--runtime'
                    mat_args = {mat_args{:}, 'runtime', varargin{ii+1}}; %#ok
                    ii = ii + 2;
                case '--threads'
                    mat_args = {mat_args{:}, 'threads', threads_string(varargin{ii+1})}; %#ok
                    ii = ii + 2;
                case '--project'
                    mat_args = {mat_args{:}, 'project', varargin{ii+1}}; %#ok
                    ii = ii + 2;
                case '--server'
                    mat_args = {mat_args{:}, 'server', true}; %#ok
                    ii = ii + 1;
                otherwise
                    decaes_args = strtrim([decaes_args, ' ', arg]); %#ok
                    ii = ii + 1;
            end
        elseif islogical(arg) || isnumeric(arg)
            arg = double(arg);
            for jj = 1:numel(arg)
                decaes_args = [decaes_args, ' ', num2str(arg(jj))]; %#ok
            end
            ii = ii + 1;
        else
            error('DECAES flag/value arguments must be char, logical, or numeric values, or arrays of such values');
        end
    end

    % Parse Matlab inputs
    p = inputParser;
    addParameter(p, 'runtime', '', @ischar);
    addParameter(p, 'threads', 'auto', @(x) strcmpi(x, 'auto') || ~isnan(check_positive_int(x)));
    addParameter(p, 'project', default_project, @ischar);
    addParameter(p, 'server', false, @islogical);
    parse(p, mat_args{:});
    opts = p.Results;

    if isempty(opts.runtime)
        opts.runtime = try_find_julia_runtime;
    end

end
