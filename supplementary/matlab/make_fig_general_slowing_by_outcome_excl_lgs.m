function results = make_fig_general_slowing_by_outcome_excl_lgs(varargin)
%MAKE_FIG_GENERAL_SLOWING_BY_OUTCOME_EXCL_LGS
% Box + swarm: slowing and spike rate by 2-year recurrence and seizure control
% (General), excluding LGS (de-identified "Other Specified Generalized").
%
% Panels A/B: % slowing (index EEG for recurrence; mean EEG for sz control).
% Panels C/D: log10(spike rate/hr + 0.1) (same EEG aggregation per outcome).
%
% Wilcoxon rank-sum (Mann-Whitney) with Bonferroni across all four panels.
% Mean + 95% bootstrap CI overlay on each group.
%
% Usage:
%   make_fig_general_slowing_by_outcome_excl_lgs()
%   make_fig_general_slowing_by_outcome_excl_lgs('SavePng', true)

thisDir = fileparts(mfilename('fullpath'));
defaultOut = fullfile(thisDir, '..', 'output');

EXCL_DATA_LABEL = "Other Specified Generalized";
EXCL_DISPLAY_LABEL = "LGS";

p = inputParser;
p.addParameter('ProjectRoot', default_project_root(), @(x) ischar(x) || isstring(x));
p.addParameter('SavePng', false, @islogical);
p.addParameter('OutPng', '', @(x) ischar(x) || isstring(x));
p.addParameter('ExcludeSubtype', EXCL_DATA_LABEL, @(x) ischar(x) || isstring(x));
p.addParameter('nBoot', 5000, @isnumeric);
p.addParameter('alpha', 0.05, @isnumeric);
p.addParameter('FontSize', 20, @isnumeric);
p.parse(varargin{:});
opts = p.Results;

rng(1);

excludeSubtype = strtrim(string(opts.ExcludeSubtype));
if excludeSubtype == EXCL_DATA_LABEL
    excludeLabel = EXCL_DISPLAY_LABEL;
else
    excludeLabel = excludeSubtype;
end

fprintf('\n========== General (excl %s): slowing + spikes by outcome ==========\n', excludeLabel);

spikeCsv = fullfile(opts.ProjectRoot, 'spike_counts.csv');
reportCsv = fullfile(opts.ProjectRoot, ...
    'Routineeegpec-Deidreport_DATA_LABELS_2025-12-12_1339 (2).csv');
slowDir = fullfile(opts.ProjectRoot, 'morgoth', 'results', ...
    'light_SLOWING_General', 'pred_SLOWING_1sStep');

PatTbl = build_general_patient_cohort(spikeCsv, reportCsv, slowDir);
fprintf('Full General cohort: %d patients\n', height(PatTbl));

spec = strtrim(string(PatTbl.EpilepsySpecific));
nExcl = sum(spec == excludeSubtype);
PatTbl = PatTbl(spec ~= excludeSubtype, :);
fprintf('Excluded %s: n=%d; remaining n=%d\n', excludeLabel, nExcl, height(PatTbl));

[FigOut, stats] = plot_slowing_by_outcome(PatTbl, opts);
print_outcome_stats(stats, opts, excludeLabel);

show_figure_if_interactive(FigOut);

if opts.SavePng
    if strlength(string(opts.OutPng)) > 0
        outPath = char(opts.OutPng);
    else
        outPath = fullfile(char(defaultOut), 'FigGeneralSlowing_byOutcome_exclLGS.png');
    end
    if ~exist(fileparts(outPath), 'dir'), mkdir(fileparts(outPath)); end
    exportgraphics(FigOut, outPath, 'Resolution', 300, 'BackgroundColor', 'white');
    fprintf('Saved figure: %s\n', outPath);
end

results = struct('Figure', FigOut, 'Stats', stats, 'PatientTable', PatTbl, ...
    'ExcludedSubtype', excludeLabel, 'NExcluded', nExcl);
end

%% =====================================================================
function [FigOut, stats] = plot_slowing_by_outcome(PatTbl, opts)
fontL = opts.FontSize;

panels = {
    'A. % windows with slowing by 2-year recurrence', ...
        'seizure_recurrence_2yr', {'No recurrence', 'Recurrence'}, [0, 1], ...
        'pct_slowing', '% windows (focal + generalized)';
    'B. % windows with slowing by seizure control (MeanSzFreq)', ...
        'sz_freq_class', {'Moderate control', 'Poor control'}, [0, 1], ...
        'pct_slowing_mean', '% windows (focal + generalized)';
    'C. log10(spike rate + 0.1) by 2-year recurrence', ...
        'seizure_recurrence_2yr', {'No recurrence', 'Recurrence'}, [0, 1], ...
        'spike_rate_log', 'log10(spike rate/hr + 0.1)';
    'D. log10(spike rate + 0.1) by seizure control (MeanSzFreq)', ...
        'sz_freq_class', {'Moderate control', 'Poor control'}, [0, 1], ...
        'spike_rate_log_mean', 'log10(spike rate/hr + 0.1)';
    };

FigOut = figure('Color', 'w', 'Position', [40 40 2000 1500], ...
    'Name', 'General slowing + spikes by outcome (excl LGS)');
% Extra-large gutters between panels (easy to crop/tighten in Google Docs).
axPos = {
    [0.12 0.72 0.26 0.20], [0.66 0.72 0.26 0.20], ...
    [0.12 0.08 0.26 0.20], [0.66 0.08 0.26 0.20]};

panelKeys = {'recurrence_slowing', 'control_slowing', 'recurrence_spike', 'control_spike'};

rawPs = nan(size(panels, 1), 1);
stats = struct();
axList = gobjects(size(panels, 1), 1);

for pi = 1:size(panels, 1)
    titleStr = panels{pi, 1};
    colName = panels{pi, 2};
    grpLabels = panels{pi, 3};
    grpVals = panels{pi, 4};
    yCol = panels{pi, 5};
    yLabelStr = panels{pi, 6};

    sub = PatTbl(isfinite(PatTbl.(colName)) & isfinite(PatTbl.(yCol)), :);
    Y = sub.(yCol);
    grp = double(sub.(colName));

    grpStr = strings(height(sub), 1);
    grpStr(grp == grpVals(1)) = grpLabels{1};
    grpStr(grp == grpVals(2)) = grpLabels{2};
    G = categorical(grpStr, grpLabels);

    n0 = sum(grp == grpVals(1));
    n1 = sum(grp == grpVals(2));
    x0 = Y(grp == grpVals(1));
    x1 = Y(grp == grpVals(2));
    rawPs(pi) = ranksum(x0, x1, 'method', 'approx');

    ax = axes('Parent', FigOut, 'Position', axPos{pi}); %#ok<LAXES>
    axList(pi) = ax;
    hold(ax, 'on'); box(ax, 'off'); grid(ax, 'on');

    boxchart(ax, G, Y, 'BoxFaceAlpha', 0.25, 'MarkerStyle', 'none');
    sw = swarmchart(ax, G, Y, 14, 'filled', 'MarkerFaceAlpha', 0.22);
    sw.XJitterWidth = 0.18;

    for k = 1:2
        xg = Y(grp == grpVals(k));
        [mu, lo, hi] = bootstrap_mean_ci(xg, opts.nBoot, opts.alpha);
        add_mean_ci_overlay(ax, k, mu, lo, hi);
    end

    ylabel(ax, yLabelStr, 'FontSize', fontL);
    set(ax, 'FontSize', fontL);

    ax.XTickLabel = {sprintf('%s (N=%d)', grpLabels{1}, n0), ...
                     sprintf('%s (N=%d)', grpLabels{2}, n1)};
    ax.XTickLabelRotation = 25;

    stats.(panelKeys{pi}) = struct( ...
        'outcome', colName, 'yCol', yCol, 'title', titleStr, ...
        'labels', {grpLabels}, 'n', [n0, n1], ...
        'mean', [mean(x0, 'omitnan'), mean(x1, 'omitnan')], ...
        'median', [median(x0, 'omitnan'), median(x1, 'omitnan')], ...
        'p_raw', rawPs(pi));
end

nTests = numel(rawPs);
pBonf = min(rawPs * nTests, 1);

for pi = 1:size(panels, 1)
    colName = panels{pi, 2};
    yCol = panels{pi, 5};
    stats.(panelKeys{pi}).p_bonf = pBonf(pi);

    ax = axList(pi);

    sub = PatTbl(isfinite(PatTbl.(colName)) & isfinite(PatTbl.(yCol)), :);
    Y = sub.(yCol);
    yMaxData = max(Y(isfinite(Y)));
    yMinData = min(Y(isfinite(Y)));
    yMin = min(0, yMinData);
    dataSpan = max(yMaxData - yMin, 0.25 * max(abs(yMaxData), 1));
    yBar = yMaxData + 0.14 * dataSpan;
    tickH = 0.025 * dataSpan;
    textH = 0.055 * dataSpan;

    add_sigbar(ax, 1, 2, yBar, char(p_label(pBonf(pi))), tickH, textH);
    ylim(ax, [yMin, max(yBar + textH + 0.12 * dataSpan, yMaxData + 0.08 * dataSpan)]);
end
end

function print_outcome_stats(stats, opts, excludeLabel)
fprintf('\n=== General (excl %s): slowing + spikes by outcome ===\n', excludeLabel);
fprintf('Wilcoxon rank-sum; Bonferroni across 4 panels (alpha=%.2f)\n', opts.alpha);

panelKeys = {'recurrence_slowing', 'control_slowing', 'recurrence_spike', 'control_spike'};
for i = 1:numel(panelKeys)
    s = stats.(panelKeys{i});
    fprintf('\n--- %s ---\n', s.title);
    if contains(s.yCol, 'pct')
        fprintf('  %s: N=%d, mean=%.1f%%, median=%.1f%%\n', ...
            s.labels{1}, s.n(1), s.mean(1), s.median(1));
        fprintf('  %s: N=%d, mean=%.1f%%, median=%.1f%%\n', ...
            s.labels{2}, s.n(2), s.mean(2), s.median(2));
    else
        fprintf('  %s: N=%d, mean=%.2f, median=%.2f\n', ...
            s.labels{1}, s.n(1), s.mean(1), s.median(1));
        fprintf('  %s: N=%d, mean=%.2f, median=%.2f\n', ...
            s.labels{2}, s.n(2), s.mean(2), s.median(2));
    end
    fprintf('  p_bonf=%s (raw p=%s)\n', char(p_label(s.p_bonf)), char(p_label(s.p_raw)));
end
end

function show_figure_if_interactive(fig)
if isempty(fig) || ~isvalid(fig), return; end
set(fig, 'Visible', 'on');
if usejava('desktop') && ~batchStartupOptionUsed
    figure(fig);
    drawnow;
end
end

%% =====================================================================
%  Cohort + shared helpers (aligned with run_slowing_figures.m)
%% =====================================================================

function root = default_project_root()
thisFile = mfilename('fullpath');
root = fileparts(fileparts(fileparts(thisFile)));
end

function PatTbl = build_general_patient_cohort(spikeCsv, reportCsv, slowDir)
MAX_ROUTINE_HOURS   = 4;
POOR_CUTOFF_PER_MO  = 4.33;
NESD_LABEL          = "non-epileptic seizure disorder";
badTypes            = lower(["uncertain if epilepsy","unknown or mrn not found",""]);
allowable_visits    = [
    "CONSULT VISIT","ESTABLISHED PATIENT VISIT",...
    "FOLLOW-UP PATIENT CLINIC","NEW PATIENT CLINIC","NEW PATIENT VISIT",...
    "NPV MANAGEMENT DURING COVID-19","NPV NEUROLOGY",...
    "RETURN ANNUAL VISIT","RETURN PATIENT EXTENDED","RETURN PATIENT VISIT",...
    "RPV MANAGEMENT DURING COVID-19","TELEHEALTH VIDEO VISIT RETURN"
];

SpikeTbl = readtable(spikeCsv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
ReportTbl = readtable(reportCsv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
SpikeTbl.Patient = double(SpikeTbl.Patient);
SpikeTbl.Session = double(SpikeTbl.Session);
SpikeTbl.Duration_sec = double(SpikeTbl.Duration_sec);
SpikeTbl.('count_0_46') = double(SpikeTbl.('count_0_46'));
SpikeTbl.spike_rate_per_hour = SpikeTbl.('count_0_46') ./ SpikeTbl.Duration_sec * 3600;
badSpikeDurAll = ~isfinite(SpikeTbl.Duration_sec) | SpikeTbl.Duration_sec <= 0;
SpikeTbl.spike_rate_per_hour(badSpikeDurAll) = NaN;
ReportTbl.patient_id = double(ReportTbl.patient_id);
ReportTbl.session_number = double(ReportTbl.session_number);

ReportTbl = filter_visit_arrays_by_type(ReportTbl, allowable_visits);
[SpikeTbl, ReportTbl] = filter_outpatient_routine(SpikeTbl, ReportTbl, MAX_ROUTINE_HOURS);

PatTyping = build_patient_typing(ReportTbl);
validPats = valid_epilepsy_patients(PatTyping, NESD_LABEL, badTypes);

ReportSlim = ReportTbl(ismember(ReportTbl.patient_id, validPats), :);
ReportSlim = ReportSlim(:, {'patient_id','session_number','visit_type', ...
    'visit_dates_deid','sz_freqs','visit_hasSz','start_time_deid'});
ReportSlim.Properties.VariableNames = {'Patient','Session','visit_type', ...
    'visit_dates_deid','sz_freqs','visit_hasSz','start_time_deid'};

SessTbl = innerjoin(SpikeTbl(:, {'Patient','Session','Duration_sec','spike_rate_per_hour'}), ...
    ReportSlim, 'Keys', {'Patient','Session'});
SessTbl = innerjoin(SessTbl, PatTyping(:, {'Patient','EpiType3'}), 'Keys', 'Patient');
SessTbl = SessTbl(SessTbl.EpiType3 == "General", :);

genPids = unique(SessTbl.Patient);
RepGen = ReportTbl(ismember(ReportTbl.patient_id, genPids), :);
Vuniq = build_visit_level_table_r1(RepGen);
PatSz = build_patient_mean_sz_freq(Vuniq, POOR_CUTOFF_PER_MO);

SlowSess = load_session_slowing(slowDir);
n_sess_all = height(SessTbl);
n_pats_before = numel(unique(SessTbl.Patient));
SessTblAll = SessTbl;
SessTbl = select_first_index_sessions(SessTbl, SlowSess);
n_dropped = n_pats_before - numel(unique(SessTbl.Patient));
fprintf('  Index EEG (earliest valid session): %d patients from %d sessions (%d dropped, no valid session)\n', ...
    height(SessTbl), n_sess_all, n_dropped);

PatTbl = build_pat_tbl_from_index_sessions(SessTbl, SlowSess, PatSz);
PatTbl = append_mean_eeg_per_patient(PatTbl, SessTblAll, SlowSess);
PatTbl = innerjoin(PatTbl, PatTyping(:, {'Patient','EpilepsySpecific'}), 'Keys', 'Patient');
end

function SessIdx = select_first_index_sessions(SessTbl, SlowSess)
SessTbl = sortrows(SessTbl, {'Patient','start_time_deid','Session'});
patients = unique(SessTbl.Patient);
picked = SessTbl([], :);
for k = 1:numel(patients)
    pid = patients(k);
    rows = SessTbl(SessTbl.Patient == pid, :);
    for i = 1:height(rows)
        if session_is_valid_index(rows(i, :), SlowSess)
            picked = [picked; rows(i, :)]; %#ok<AGROW>
            break;
        end
    end
end
SessIdx = picked;
end

function ok = session_is_valid_index(row, SlowSess)
ok = false;
try
    dt = parse_eeg_datetime(row.start_time_deid);
    if isnat(dt), return; end
catch
    return;
end
if ~isfinite(row.spike_rate_per_hour), return; end
if ~isempty(SlowSess)
    hit = SlowSess(SlowSess.Patient == row.Patient & SlowSess.Session == row.Session, :);
    if isempty(hit), return; end
    if ~isfinite(hit.pct_gen_slowing(1)) && ~isfinite(hit.pct_foc_slowing(1)), return; end
end
ok = true;
end

function PatTbl = build_pat_tbl_from_index_sessions(SessIdx, SlowSess, PatSz)
FOLLOWUP_DAYS = 730;
SessIdx.seizure_recurrence_2yr = nan(height(SessIdx), 1);
for i = 1:height(SessIdx)
    SessIdx.seizure_recurrence_2yr(i) = recurrence_within_window(SessIdx(i, :), FOLLOWUP_DAYS);
end
SlowSess.pct_slowing = SlowSess.pct_gen_slowing + SlowSess.pct_foc_slowing;
PatSlow = innerjoin( ...
    SlowSess(:, {'Patient','Session','pct_slowing'}), ...
    SessIdx(:, {'Patient','Session'}));
PatTbl = innerjoin(PatSlow, PatSz, 'Keys', 'Patient');
PatTbl = innerjoin(PatTbl, SessIdx(:, {'Patient','Session','seizure_recurrence_2yr','spike_rate_per_hour'}), ...
    'Keys', {'Patient','Session'});
PatTbl.spike_rate_log = log10(PatTbl.spike_rate_per_hour + 0.1);
PatTbl = PatTbl(isfinite(PatTbl.pct_slowing), :);
end

function PatTbl = append_mean_eeg_per_patient(PatTbl, SessTblAll, SlowSess)
LOG_EPS = 0.1;
SlowSess.pct_slowing = SlowSess.pct_gen_slowing + SlowSess.pct_foc_slowing;
featSess = innerjoin( ...
    SlowSess(:, {'Patient','Session','pct_slowing'}), ...
    SessTblAll(:, {'Patient','Session','spike_rate_per_hour'}), ...
    'Keys', {'Patient','Session'});
validMask = false(height(SessTblAll), 1);
for i = 1:height(SessTblAll)
    validMask(i) = session_is_valid_index(SessTblAll(i, :), SlowSess);
end
featValid = featSess(validMask, :);
[gp, pids] = findgroups(featValid.Patient);
PatMean = table( ...
    pids, ...
    splitapply(@(x) mean(x, 'omitnan'), featValid.pct_slowing, gp), ...
    splitapply(@(x) mean(x, 'omitnan'), featValid.spike_rate_per_hour, gp), ...
    splitapply(@numel, featValid.pct_slowing, gp), ...
    'VariableNames', {'Patient','pct_slowing_mean','spike_rate_per_hour_mean', ...
    'n_eeg_sessions_used'});
PatTbl = outerjoin(PatTbl, PatMean, 'Keys', 'Patient', 'MergeKeys', true);
PatTbl.spike_rate_log_mean = log10(PatTbl.spike_rate_per_hour_mean + LOG_EPS);
end

function Vuniq = build_visit_level_table_r1(ReportTbl)
PV = table('Size', [0 4], 'VariableTypes', {'double','datetime','double','double'}, ...
    'VariableNames', {'Patient','VisitDate','Freq','HasSz'});
for j = 1:height(ReportTbl)
    pid = double(ReportTbl.patient_id(j));
    dates = parse_json_strings(ReportTbl.visit_dates_deid(j));
    freqs = parse_json_doubles(ReportTbl.sz_freqs(j));
    haszs = parse_json_doubles(ReportTbl.visit_hasSz(j));
    if isempty(dates) || ~(numel(freqs) == numel(dates) && numel(haszs) == numel(dates))
        continue;
    end
    for k = 1:numel(dates)
        try
            vdt = datetime(strtrim(string(dates{k})), 'InputFormat', 'yyyy-MM-dd');
            fq = freqs(k);
            if ~isfinite(fq) || fq < 0, fq = NaN; end
            hz = haszs(k);
            if hz == 2, hz = NaN; end
            PV = [PV; table(pid, vdt, fq, hz, 'VariableNames', PV.Properties.VariableNames)]; %#ok<AGROW>
        catch
        end
    end
end
if height(PV) == 0
    Vuniq = table('Size', [0 4], 'VariableTypes', {'double','datetime','double','double'}, ...
        'VariableNames', {'Patient','VisitDate','Freq','HasSz'});
    return;
end
[gv, pid_keys, date_keys] = findgroups(PV.Patient, PV.VisitDate);
Freq_agg = splitapply(@(x) mean(x(isfinite(x)), 'omitnan'), PV.Freq, gv);
Has_agg  = splitapply(@max_hasSz, PV.HasSz, gv);
Vuniq = table(pid_keys, date_keys, Freq_agg, Has_agg, ...
    'VariableNames', {'Patient','VisitDate','Freq','HasSz'});
Vuniq.Freq_R1 = Vuniq.Freq;
mask_rule1 = ~isfinite(Vuniq.Freq_R1) & (Vuniq.HasSz == 0);
Vuniq.Freq_R1(mask_rule1) = 0;
end

function PatSz = build_patient_mean_sz_freq(Vuniq, poorCutoff)
if isempty(Vuniq) || height(Vuniq) == 0
    PatSz = table('Size', [0 3], 'VariableTypes', {'double','double','double'}, ...
        'VariableNames', {'Patient','mean_sz_freq','sz_freq_class'});
    return;
end
[gp, pids] = findgroups(Vuniq.Patient);
MeanSzFreq = splitapply(@(x) mean(x, 'omitnan'), Vuniq.Freq_R1, gp);
PatSz = table(pids, MeanSzFreq, 'VariableNames', {'Patient','mean_sz_freq'});
PatSz.sz_freq_class = nan(height(PatSz), 1);
ok = isfinite(PatSz.mean_sz_freq);
PatSz.sz_freq_class(ok) = double(PatSz.mean_sz_freq(ok) > poorCutoff);
end

function out = max_hasSz(x)
x = x(isfinite(x));
if isempty(x), out = NaN; else, out = max(x); end
end

function SlowSess = load_session_slowing(slowDirs)
if ischar(slowDirs) || isstring(slowDirs), slowDirs = {char(slowDirs)}; end
allRaw = table('Size', [0 6], ...
    'VariableTypes', {'double','double','double','double','double','double'}, ...
    'VariableNames', {'Patient','Session','pct_no','pct_foc','pct_gen','n_win'});
for d = 1:numel(slowDirs)
    slowDir = slowDirs{d};
    files = dir(fullfile(slowDir, '*.csv'));
    if isempty(files), warning('No slowing CSV files in %s', slowDir); continue; end
    nF = numel(files);
    Patient = nan(nF, 1); Session = nan(nF, 1);
    pct_no = nan(nF, 1); pct_foc = nan(nF, 1); pct_gen = nan(nF, 1); n_win = nan(nF, 1);
    for i = 1:nF
        tok = regexp(files(i).name, 'sub-Penn(\d+)_ses-(\d+)', 'tokens', 'once');
        if isempty(tok), continue; end
        Patient(i) = str2double(tok{1}); Session(i) = str2double(tok{2});
        T = readtable(fullfile(slowDir, files(i).name));
        cls = T.pred_class; n = numel(cls); n_win(i) = n;
        if n == 0
            pct_no(i) = NaN; pct_foc(i) = NaN; pct_gen(i) = NaN;
        else
            pct_no(i) = 100 * mean(cls == 0);
            pct_foc(i) = 100 * mean(cls == 1);
            pct_gen(i) = 100 * mean(cls == 2);
        end
    end
    Raw = table(Patient, Session, pct_no, pct_foc, pct_gen, n_win);
    Raw = Raw(isfinite(Raw.Patient), :);
    allRaw = [allRaw; Raw]; %#ok<AGROW>
end
if height(allRaw) == 0, error('No slowing session data found.'); end
[gs, pid, ses] = findgroups(allRaw.Patient, allRaw.Session);
pct_gen_agg = splitapply(@weighted_mean, allRaw.pct_gen, allRaw.n_win, gs);
pct_foc_agg = splitapply(@weighted_mean, allRaw.pct_foc, allRaw.n_win, gs);
SlowSess = table(pid, ses, pct_gen_agg, pct_foc_agg, ...
    'VariableNames', {'Patient','Session','pct_gen_slowing','pct_foc_slowing'});
end

function m = weighted_mean(x, w)
ok = isfinite(x) & isfinite(w) & w > 0;
if ~any(ok), m = NaN; return; end
m = sum(x(ok) .* w(ok)) / sum(w(ok));
end

function y = recurrence_within_window(row, followupDays)
eeg_dt = parse_eeg_datetime(row.start_time_deid);
if isnat(eeg_dt), y = NaN; return; end
dates = parse_json_strings(row.visit_dates_deid);
haszs = parse_json_doubles(row.visit_hasSz);
if isempty(dates) || numel(haszs) ~= numel(dates), y = NaN; return; end
post = [];
for k = 1:numel(dates)
    try
        vdt = datetime(strtrim(string(dates{k})), 'InputFormat', 'yyyy-MM-dd');
        gap = days(vdt - eeg_dt);
        if gap > 0 && gap <= followupDays
            hz = haszs(k);
            if hz == 2, hz = NaN; end
            if isfinite(hz), post(end+1) = hz; end %#ok<AGROW>
        end
    catch
    end
end
if isempty(post), y = NaN; else, y = double(any(post == 1)); end
end

function dt = parse_eeg_datetime(val)
if isdatetime(val), dt = val; if numel(dt) > 1, dt = dt(1); end; return; end
s = strtrim(string(val));
if ismissing(s) || s == "", dt = NaT; return; end
try, dt = datetime(s, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
catch
    try, dt = datetime(s);
    catch, dt = NaT;
    end
end
end

function validPats = valid_epilepsy_patients(PatTyping, nesdLabel, badTypes)
etype = lower(strtrim(string(PatTyping.EpilepsyType)));
isNesd = etype == lower(strtrim(nesdLabel));
isBad = ismember(etype, badTypes) | ismissing(etype) | strlength(etype) == 0;
validPats = PatTyping.Patient(~isNesd & ~isBad);
end

function T = build_patient_typing(ReportTbl)
[gp, pids] = findgroups(double(ReportTbl.patient_id));
etype_one = splitapply(@first_nonempty_string, ReportTbl.epilepsy_type, gp);
espec_one = splitapply(@first_nonempty_string, ReportTbl.epilepsy_specific, gp);
T = table(pids, etype_one, espec_one, ...
    'VariableNames', {'Patient','EpilepsyType','EpilepsySpecific'});
spec_norm = lower(strtrim(string(T.EpilepsySpecific)));
type_norm = lower(strtrim(string(T.EpilepsyType)));
E3 = strings(height(T), 1);
E3(contains(spec_norm, "temporal")) = "Temporal";
E3(contains(spec_norm, "frontal"))  = "Frontal";
E3(strlength(E3) == 0 & type_norm == "general") = "General";
T.EpiType3 = categorical(E3, ["General","Temporal","Frontal"]);
end

function s = first_nonempty_string(x)
x = strtrim(string(x));
x = x(~ismissing(x) & strlength(x) > 0);
if isempty(x), s = ""; else, s = x(1); end
end

function [S, R] = filter_outpatient_routine(S, R, maxHours)
acq = lower(strtrim(string(R.acquired_on)));
cls = lower(strtrim(string(R.report_PATIENT_CLASS)));
jay = lower(strtrim(string(R.jay_in_or_out)));
isOut = contains(acq, "spe") | contains(acq, "radnor") | ...
    (cls == "outpatient") | (jay == "out");
OutKeys = unique(R(isOut, {'patient_id','session_number'}));
OutKeys.Properties.VariableNames = {'Patient','Session'};
isRoutine = isfinite(S.Duration_sec) & S.Duration_sec <= maxHours * 3600;
RoutineKeys = unique(S(isRoutine, {'Patient','Session'}));
KeepKeys = innerjoin(OutKeys, RoutineKeys);
S = innerjoin(S(:, {'Patient','Session','Duration_sec','spike_rate_per_hour'}), KeepKeys);
R = innerjoin(R, KeepKeys, 'LeftKeys', {'patient_id','session_number'}, ...
    'RightKeys', {'Patient','Session'});
end

function R = filter_visit_arrays_by_type(R, allowable_visits)
for i = 1:height(R)
    vt = parse_json_strings(R.visit_type(i));
    dates = parse_json_strings(R.visit_dates_deid(i));
    szf = parse_json_doubles(R.sz_freqs(i));
    hs  = parse_json_doubles(R.visit_hasSz(i));
    if isempty(vt) || ~(numel(vt) == numel(dates) && numel(vt) == numel(szf) && numel(vt) == numel(hs))
        R.visit_type(i) = "[]"; R.visit_dates_deid(i) = "[]";
        R.sz_freqs(i) = "[]"; R.visit_hasSz(i) = "[]";
        continue;
    end
    keep = ismember(string(vt), allowable_visits);
    if ~any(keep)
        R.visit_type(i) = "[]"; R.visit_dates_deid(i) = "[]";
        R.sz_freqs(i) = "[]"; R.visit_hasSz(i) = "[]";
        continue;
    end
    R.visit_type(i) = jsonencode(vt(keep));
    R.visit_dates_deid(i) = jsonencode(dates(keep));
    R.sz_freqs(i) = jsonencode(szf(keep));
    R.visit_hasSz(i) = jsonencode(hs(keep));
end
end

function arr = parse_json_strings(s)
s = strtrim(string(s));
if s == "" || s == "[]" || s == "null" || s == "[null]", arr = {}; return; end
dec = jsondecode(char(s));
if iscell(dec), arr = cellfun(@char, dec, 'UniformOutput', false);
elseif isstring(dec) || ischar(dec), arr = {char(dec)};
else, arr = cellstr(string(dec));
end
end

function arr = parse_json_doubles(s)
s = strtrim(string(s));
if s == "" || s == "[]" || s == "null" || s == "[null]", arr = []; return; end
s = regexprep(s, 'null', 'NaN', 'ignorecase');
arr = double(jsondecode(char(s)));
arr = arr(:);
end

function [mu, lo, hi] = bootstrap_mean_ci(x, nBoot, alpha)
x = double(x(:)); x = x(isfinite(x));
if isempty(x), mu = NaN; lo = NaN; hi = NaN; return; end
mu = mean(x);
if nBoot <= 0 || numel(x) < 2, lo = NaN; hi = NaN; return; end
n = numel(x);
bootM = nan(nBoot, 1);
for b = 1:nBoot
    bootM(b) = mean(x(randi(n, n, 1)));
end
lo = prctile(bootM, 100 * (alpha / 2));
hi = prctile(bootM, 100 * (1 - alpha / 2));
end

function add_mean_ci_overlay(ax, xpos, mu, lo, hi)
if ~isfinite(mu), return; end
if isfinite(lo) && isfinite(hi)
    plot(ax, [xpos xpos], [lo hi], 'k-', 'LineWidth', 3);
end
plot(ax, xpos, mu, 'kd', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
end

function add_sigbar(ax, x1, x2, y, ptext, tickH, textH)
if nargin < 6, tickH = 0.03 * diff(ax.YLim); end
if nargin < 7, textH = 0.04 * diff(ax.YLim); end
plot(ax, [x1 x1 x2 x2], [y - tickH, y, y, y - tickH], 'k-', 'LineWidth', 1.3);
text(ax, mean([x1 x2]), y + textH, ptext, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14);
end

function pStr = p_label(p)
if isnan(p), pStr = "p=NaN"; return; end
if p < 0.001, pStr = "p<0.001"; return; end
if p < 0.01, pStr = sprintf("p=%.2g", p); return; end
pStr = sprintf("p=%.2f", p);
end
