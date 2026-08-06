function results = make_fig_general_slowing_by_subtype(varargin)
%MAKE_FIG_GENERAL_SLOWING_BY_SUBTYPE
% Box + swarm plot of % slowing by de-identified generalized epilepsy subtype.
%
% Generalized epilepsy patients only (EpiType3 == General). One row per patient
% at the earliest valid index EEG (matches run_slowing_figures.m / Python ML).
% Y-axis: pct_slowing = pct_foc_slowing + pct_gen_slowing (% Morgoth windows).
%
% Pairwise Mann-Whitney tests with Bonferroni correction across all subtype pairs
% (groups with n >= MinGroupN only). Full pairwise table returned in stats.Pairwise.
%
% Usage:
%   make_fig_general_slowing_by_subtype()
%   make_fig_general_slowing_by_subtype('SavePng', true)
%
% Requires Statistics and Machine Learning Toolbox.

thisDir = fileparts(mfilename('fullpath'));
defaultOut = fullfile(thisDir, '..', 'output');

p = inputParser;
p.addParameter('ProjectRoot', default_project_root(), @(x) ischar(x) || isstring(x));
p.addParameter('SavePng', false, @islogical);
p.addParameter('OutPng', fullfile(defaultOut, 'FigGeneralSlowing_bySubtype_4groups_bonf.png'), ...
    @(x) ischar(x) || isstring(x));
p.addParameter('MinGroupN', 5, @isnumeric);
p.addParameter('nBoot', 5000, @isnumeric);
p.addParameter('alpha', 0.05, @isnumeric);
p.addParameter('FontSize', 20, @isnumeric);
p.addParameter('ReferenceGroup', "", @(x) ischar(x) || isstring(x));
p.addParameter('SortBy', 'Median', @(x) ischar(x) || isstring(x));
p.addParameter('SigBarStep', 16, @isnumeric);  % vertical gap between sig brackets (% units)
% Keep only these mapped subtype labels (Aim2 KEEP set used for AUC subtype figs)
p.addParameter('KeepSubtypes', [ ...
    "Unspecified Generalized", ...
    "Juvenile Myoclonic", ...
    "GTCA", ...
    "Other Specified Generalized (LGS)"], @(x) isstring(x) || iscellstr(x) || ischar(x));
p.addParameter('DisplayLabels', [ ...
    "Unspecified", "JME", "GTCA", "LGS"], @(x) isstring(x) || iscellstr(x) || ischar(x));
p.parse(varargin{:});
opts = p.Results;
rng(1);

PatTbl = build_general_subtype_cohort(opts.ProjectRoot);
keep = string(opts.KeepSubtypes);
PatTbl = PatTbl(ismember(string(PatTbl.SubtypeLabel), keep), :);
fprintf('Generalized epilepsy (index EEG, slowing available, kept subtypes): %d patients\n', height(PatTbl));
for k = 1:numel(keep)
    fprintf('  %s: N=%d\n', keep(k), sum(string(PatTbl.SubtypeLabel) == keep(k)));
end

[FigOut, stats] = plot_slowing_by_subtype(PatTbl, opts);
print_subtype_stats(stats, opts);

if opts.SavePng
    outPath = char(opts.OutPng);
    if ~exist(fileparts(outPath), 'dir'), mkdir(fileparts(outPath)); end
    exportgraphics(FigOut, outPath, 'Resolution', 300, 'BackgroundColor', 'white');
    fprintf('Saved figure: %s\n', outPath);
end

if usejava('desktop') && ~batchStartupOptionUsed
    set(FigOut, 'Visible', 'on');
end

results = struct('Figure', FigOut, 'Stats', stats, 'PatientTable', PatTbl);
end

%% =====================================================================
function PatTbl = build_general_subtype_cohort(projectRoot)
% Index EEG cohort aligned with generalized_ml.py (723 patients with demographics).
slowDir = fullfile(projectRoot, 'morgoth', 'results', ...
    'light_SLOWING_General', 'pred_SLOWING_1sStep');
spikeCsv = fullfile(projectRoot, 'spike_counts.csv');
reportCsv = fullfile(projectRoot, ...
    'Routineeegpec-Deidreport_DATA_LABELS_2025-12-12_1339 (2).csv');

MAX_ROUTINE_HOURS = 4;
NESD_LABEL = "non-epileptic seizure disorder";
badTypes = lower(["uncertain if epilepsy","unknown or mrn not found",""]);
allowable_visits = [
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
SpikeTbl.spike_rate_per_hour(~isfinite(SpikeTbl.Duration_sec) | SpikeTbl.Duration_sec <= 0) = NaN;
ReportTbl.patient_id = double(ReportTbl.patient_id);
ReportTbl.session_number = double(ReportTbl.session_number);

ReportTbl = filter_visit_arrays_by_type(ReportTbl, allowable_visits);
[SpikeTbl, ReportTbl] = filter_outpatient_routine(SpikeTbl, ReportTbl, MAX_ROUTINE_HOURS);

PatTyping = build_patient_typing(ReportTbl);
validPats = valid_epilepsy_patients(PatTyping, NESD_LABEL, badTypes);
PatTyping = PatTyping(ismember(PatTyping.Patient, validPats), :);

ReportSlim = ReportTbl(ismember(ReportTbl.patient_id, validPats), :);
ReportSlim = ReportSlim(:, {'patient_id','session_number','start_time_deid', ...
    'deid_birth_date','nlp_gender'});
ReportSlim.Properties.VariableNames = {'Patient','Session','start_time_deid', ...
    'deid_birth_date','nlp_gender'};

SessTbl = innerjoin(SpikeTbl(:, {'Patient','Session','spike_rate_per_hour'}), ...
    ReportSlim, 'Keys', {'Patient','Session'});
SessTbl = innerjoin(SessTbl, PatTyping(:, {'Patient','EpiType3'}), 'Keys', 'Patient');
SessTbl = SessTbl(SessTbl.EpiType3 == "General", :);
SessTbl.age_at_eeg = nan(height(SessTbl), 1);
SessTbl.gender = nan(height(SessTbl), 1);
for i = 1:height(SessTbl)
    SessTbl.age_at_eeg(i) = compute_age_at_eeg(SessTbl(i, :));
    SessTbl.gender(i) = compute_gender(SessTbl(i, :));
end

SlowSess = load_session_slowing(slowDir);
SessIdx = select_first_index_sessions(SessTbl, SlowSess);

SlowSess.pct_slowing = SlowSess.pct_gen_slowing + SlowSess.pct_foc_slowing;
PatSlow = innerjoin( ...
    SlowSess(:, {'Patient','Session','pct_slowing'}), ...
    SessIdx(:, {'Patient','Session','age_at_eeg','gender'}), ...
    'Keys', {'Patient','Session'});

PatSubtype = build_patient_subtype_table(ReportTbl);
PatTbl = innerjoin(PatSlow, PatSubtype(:, {'Patient','SubtypeLabel'}), 'Keys', 'Patient');
PatTbl = PatTbl(isfinite(PatTbl.pct_slowing) & isfinite(PatTbl.age_at_eeg) & ...
    isfinite(PatTbl.gender) & strlength(strtrim(PatTbl.SubtypeLabel)) > 0, :);
end

function PatSubtype = build_patient_subtype_table(ReportTbl)
Rep = ReportTbl(:, {'patient_id','epilepsy_specific'});
Rep = Rep(isfinite(Rep.patient_id), :);
[gp, pids] = findgroups(double(Rep.patient_id));
labels = strings(numel(pids), 1);
for k = 1:numel(pids)
    vals = strtrim(string(Rep.epilepsy_specific(gp == k)));
    vals = vals(~ismissing(vals) & strlength(vals) > 0);
    [raw, ~] = resolve_epilepsy_specific(vals);
    if strlength(raw) == 0
        labels(k) = missing;
    else
        labels(k) = map_general_subtype(raw);
    end
end
PatSubtype = table(pids, labels, 'VariableNames', {'Patient','SubtypeLabel'});
PatSubtype = PatSubtype(~ismissing(PatSubtype.SubtypeLabel), :);
end

function [raw, hadConflict] = resolve_epilepsy_specific(values)
raw = "";
hadConflict = false;
if isempty(values)
    return;
end
uniq = unique(values, 'stable');
if numel(uniq) == 1
    raw = uniq(1);
    return;
end
hadConflict = true;
counts = zeros(numel(uniq), 1);
for i = 1:numel(uniq)
    counts(i) = sum(values == uniq(i));
end
[sortedCounts, ord] = sort(counts, 'descend');
if numel(sortedCounts) > 1 && sortedCounts(1) == sortedCounts(2)
    raw = "";
    return;
end
raw = uniq(ord(1));
end

function label = map_general_subtype(raw)
s = lower(strtrim(string(raw)));
if ismissing(s) || strlength(s) == 0 || s == "unknown or mrn not found" || ...
        s == "unclassified or unspecified"
    label = "Unclassified";
elseif s == "unspecified generalized"
    label = "Unspecified Generalized";
elseif contains(s, "absence")
    label = "Absence";
elseif s == "juvenile myoclonic"
    label = "Juvenile Myoclonic";
elseif s == "gtca"
    label = "GTCA";
elseif s == "other specified generalized" || contains(s, "lgs")
    label = "Other Specified Generalized (LGS)";
elseif s == "combined generalized and focal"
    label = "Combined Generalized and Focal";
else
    label = "Unclassified";
end
end

function age = compute_age_at_eeg(row)
age = NaN;
try
    eegDt = parse_eeg_datetime(row.start_time_deid);
    birthDt = datetime(strtrim(string(row.deid_birth_date)), 'InputFormat', 'yyyy-MM-dd');
    if isnat(eegDt) || isnat(birthDt), return; end
    age = days(eegDt - birthDt) / 365.25;
catch
end
end

function g = compute_gender(row)
g = NaN;
s = upper(strtrim(string(row.nlp_gender)));
if s == "F", g = 0;
elseif s == "M", g = 1;
end
end

%% =====================================================================
function [FigOut, stats] = plot_slowing_by_subtype(PatTbl, opts)
%PLOT_SLOWING_BY_SUBTYPE  Box + swarm plot with full pairwise stats, reference-only annotations.
%
% Visualization choices (stats unchanged):
%   - opts.SortBy: 'Median' (default) orders x-axis by ascending median slowing; 'N' keeps
%     descending-count order for backward compatibility.
%   - opts.ReferenceGroup: empty (default) uses the largest-N subtype (stored in stats).
%   - Groups with n < MinGroupN are drawn with lower alpha and excluded from testing labels.
%   - Pairwise Mann–Whitney with Bonferroni across all eligible subtype pairs.

% Optional short display labels (aligned with KeepSubtypes order)
keep = string(opts.KeepSubtypes);
dispLab = string(opts.DisplayLabels);
labMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
if numel(dispLab) == numel(keep)
    for k = 1:numel(keep)
        labMap(char(keep(k))) = char(dispLab(k));
    end
end
dispLabels = strings(height(PatTbl), 1);
for i = 1:height(PatTbl)
    key = char(string(PatTbl.SubtypeLabel(i)));
    if isKey(labMap, key)
        dispLabels(i) = string(labMap(key));
    else
        dispLabels(i) = string(PatTbl.SubtypeLabel(i));
    end
end
PatTbl.SubtypeLabel = dispLabels;

Y = PatTbl.pct_slowing;
G = categorical(PatTbl.SubtypeLabel);

[cats, ~, ic] = unique(G);
counts = accumarray(ic, 1);
nGroups = numel(cats);

groupMeans = nan(nGroups, 1);
groupMedians = nan(nGroups, 1);
for k = 1:nGroups
    xg = Y(G == cats(k));
    groupMeans(k) = mean(xg, 'omitnan');
    groupMedians(k) = median(xg, 'omitnan');
end

% X-axis order: median ascending (default) surfaces low-slowing subtypes first for reviewers.
sortBy = upper(strtrim(string(opts.SortBy)));
if sortBy == "N"
    [~, ord] = sort(counts, 'descend');
elseif sortBy == "MEDIAN"
    [~, ord] = sort(groupMedians, 'ascend');
else
    error('SortBy must be ''N'' or ''Median''.');
end
cats = cats(ord);
counts = counts(ord);
groupMeans = groupMeans(ord);
groupMedians = groupMedians(ord);

% Reference group defaults to largest N (typically Unspecified Generalized), independent of SortBy.
refGroup = string(strtrim(opts.ReferenceGroup));
if strlength(refGroup) == 0
    [~, refPick] = max(counts);
    refGroup = string(cats(refPick));
elseif ~any(cats == refGroup)
    error('ReferenceGroup "%s" not found among subtype labels.', refGroup);
end

G = reordercats(categorical(PatTbl.SubtypeLabel), cellstr(cats));
groupCounts = counts;

% Pairwise Mann-Whitney (Bonferroni across eligible pairs) — full table, unchanged logic.
eligible = groupCounts >= opts.MinGroupN;
pairRows = table('Size', [0 5], ...
    'VariableTypes', {'string','string','double','double','logical'}, ...
    'VariableNames', {'GroupA','GroupB','p_raw','p_bonf','Significant'});

for i = 1:nGroups
    for j = i + 1:nGroups
        a = string(cats(i)); b = string(cats(j));
        xA = Y(G == a);
        xB = Y(G == b);
        if eligible(i) && eligible(j) && numel(xA) >= opts.MinGroupN && numel(xB) >= opts.MinGroupN
            pRaw = ranksum(xA, xB, 'method', 'approx');
        else
            pRaw = NaN;
        end
        pairRows = [pairRows; table(a, b, pRaw, NaN, false, ...
            'VariableNames', pairRows.Properties.VariableNames)]; %#ok<AGROW>
    end
end
nTests = sum(isfinite(pairRows.p_raw));
if nTests > 0
    pairRows.p_bonf = min(pairRows.p_raw * nTests, 1);
    pairRows.Significant = pairRows.p_bonf < opts.alpha;
end

% Global Kruskal-Wallis (eligible groups only)
if sum(eligible) >= 2
    mask = false(height(PatTbl), 1);
    for k = 1:nGroups
        if eligible(k), mask = mask | (G == cats(k)); end
    end
    [p_kw, ~, stats_kw] = kruskalwallis(Y(mask), G(mask), 'off');
else
    p_kw = NaN;
    stats_kw = struct('n', 0);
end

FigOut = figure('Color', 'w', 'Position', [80 80 1100 820]);
ax = axes(FigOut); hold(ax, 'on'); box(ax, 'off'); grid(ax, 'on');

% Uniform blue for all subtypes (MATLAB default series color #1)
faceBlue = [0 0.4470 0.7410];

% Per-group rendering so under-powered groups can be de-emphasized without affecting others.
for k = 1:nGroups
    mask = G == cats(k);
    xg = Y(mask);
    smallGroup = groupCounts(k) < opts.MinGroupN;
    if ~smallGroup && groupCounts(k) >= 2
        boxchart(ax, G(mask), xg, ...
            'BoxFaceColor', faceBlue, 'BoxFaceAlpha', 0.25, ...
            'BoxEdgeColor', faceBlue, 'MarkerStyle', 'none');
    end
    swarmAlpha = 0.10;
    if ~smallGroup
        swarmAlpha = 0.22;
    end
    sw = swarmchart(ax, G(mask), xg, 14, 'filled', ...
        'MarkerFaceColor', faceBlue, 'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', swarmAlpha);
    sw.XJitterWidth = 0.18;
    if groupCounts(k) >= 2
        [mu, lo, hi] = bootstrap_mean_ci(xg, opts.nBoot, opts.alpha);
        add_mean_ci_overlay(ax, k, mu, lo, hi);
    end
end

ylabel(ax, '% windows (focal + generalized)', 'FontSize', opts.FontSize);
xlabel(ax, 'Generalized epilepsy subtype (de-identified)', 'FontSize', opts.FontSize);
set(ax, 'FontSize', opts.FontSize);
ax.TickLabelInterpreter = 'none';
ylim(ax, [0, 100]);
ax.YTick = 0:20:100;

ax.XTickLabel = cellstr(cats);
ax.XTickLabelRotation = 25;

stats = struct();
stats.GroupLabels = cats;
stats.GroupCounts = groupCounts;
stats.GroupMeans = groupMeans;
stats.GroupMedians = groupMedians;
stats.KruskalWallis_p = p_kw;
stats.KruskalWallis = stats_kw;
stats.Pairwise = pairRows;
stats.nBonfTests = nTests;
stats.MinGroupN = opts.MinGroupN;
stats.ReferenceGroup = refGroup;
stats.SortBy = char(string(opts.SortBy));
end

%% =====================================================================
function print_subtype_stats(stats, opts)
fprintf('\n=== Generalized epilepsy subtypes: %% slowing (index EEG) ===\n');
fprintf('Kruskal-Wallis (groups with n >= %d): p = %s\n', ...
    opts.MinGroupN, char(p_label(stats.KruskalWallis_p)));
fprintf('\nSubtype summary:\n');
for k = 1:numel(stats.GroupLabels)
    fprintf('  %s: N=%d, mean=%.1f%%, median=%.1f%%\n', ...
        stats.GroupLabels(k), stats.GroupCounts(k), ...
        stats.GroupMeans(k), stats.GroupMedians(k));
end

sig = stats.Pairwise(stats.Pairwise.Significant, :);
fprintf('\nPairwise Mann-Whitney (Bonferroni across %d eligible pairs, MinGroupN=%d):\n', ...
    stats.nBonfTests, stats.MinGroupN);
if isempty(sig)
    fprintf('  No pairs significant at alpha=%.2f after Bonferroni.\n', opts.alpha);
else
    for r = 1:height(sig)
        fprintf('  %s vs %s: p_bonf=%s (raw p=%s)\n', ...
            sig.GroupA(r), sig.GroupB(r), ...
            char(p_label(sig.p_bonf(r))), char(p_label(sig.p_raw(r))));
    end
end
end

%% =====================================================================
%  Shared helpers (aligned with run_slowing_figures.m)
%% =====================================================================

function root = default_project_root()
thisFile = mfilename('fullpath');
root = fileparts(fileparts(fileparts(thisFile)));
end

function SessIdx = select_first_index_sessions(SessTbl, SlowSess)
SessTbl = sortrows(SessTbl, {'Patient','start_time_deid','Session'});
patients = unique(SessTbl.Patient);
picked = SessTbl([], :);
for k = 1:numel(patients)
    pid = patients(k);
    rows = SessTbl(SessTbl.Patient == pid, :);
    for i = 1:height(rows)
        if session_is_valid_index(rows(i, :), SlowSess, true)
            picked = [picked; rows(i, :)]; %#ok<AGROW>
            break;
        end
    end
end
SessIdx = picked;
end

function ok = session_is_valid_index(row, SlowSess, requireSlowing)
if nargin < 3, requireSlowing = true; end
ok = false;
try
    dt = parse_eeg_datetime(row.start_time_deid);
    if isnat(dt), return; end
catch
    return;
end
if ~isfinite(row.spike_rate_per_hour), return; end
if ~isfinite(row.age_at_eeg) || ~isfinite(row.gender), return; end
if requireSlowing && ~isempty(SlowSess)
    hit = SlowSess(SlowSess.Patient == row.Patient & SlowSess.Session == row.Session, :);
    if isempty(hit), return; end
    if ~isfinite(hit.pct_gen_slowing(1)) && ~isfinite(hit.pct_foc_slowing(1)), return; end
end
ok = true;
end

function SlowSess = load_session_slowing(slowDirs)
if ischar(slowDirs) || isstring(slowDirs), slowDirs = {char(slowDirs)}; end
allRaw = table('Size', [0 6], ...
    'VariableTypes', {'double','double','double','double','double','double'}, ...
    'VariableNames', {'Patient','Session','pct_no','pct_foc','pct_gen','n_win'});
for d = 1:numel(slowDirs)
    slowDir = slowDirs{d};
    files = dir(fullfile(slowDir, '*.csv'));
    if isempty(files), continue; end
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

function validPats = valid_epilepsy_patients(PatTyping, nesdLabel, badTypes)
etype = lower(strtrim(string(PatTyping.EpilepsyType)));
isNesd = etype == lower(strtrim(nesdLabel));
isBad = ismember(etype, badTypes) | ismissing(etype) | strlength(etype) == 0;
validPats = PatTyping.Patient(~isNesd & ~isBad);
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

function s = first_nonempty_string(x)
x = strtrim(string(x));
x = x(~ismissing(x) & strlength(x) > 0);
if isempty(x), s = ""; else, s = x(1); end
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
dec = jsondecode(char(s));
arr = double(dec(:));
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

function add_sigbar(ax, x1, x2, y, ptext, tickH, textH, fontSize)
if nargin < 6, tickH = 0.03 * diff(ax.YLim); end
if nargin < 7, textH = 0.04 * diff(ax.YLim); end
if nargin < 8, fontSize = 16; end
plot(ax, [x1 x1 x2 x2], [y - tickH, y, y, y - tickH], 'k-', 'LineWidth', 1.3);
text(ax, mean([x1 x2]), y + textH, ptext, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', fontSize, 'Color', [0.25 0.25 0.25], 'FontWeight', 'bold');
end

function s = stars(p)
if isnan(p), s = ""; return; end
if p < 0.001, s = "***"; return; end
if p < 0.01, s = "**"; return; end
if p < 0.05, s = "*"; return; end
s = "";
end

function pStr = p_label(p)
if isnan(p), pStr = "p=NaN"; return; end
if p < 0.001, pStr = "p<0.001"; return; end
if p < 0.01, pStr = sprintf("p=%.2g", p); return; end
pStr = sprintf("p=%.2f", p);
end
