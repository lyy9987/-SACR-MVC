function native_correct_cifar_seeds(dataset,seeds)
work=fileparts(mfilename('fullpath'));root=fileparts(fileparts(fileparts(work)));
source=fullfile(root,'_teammate_all_datasets',char([48 51 95 35770 25991 22797 29616 20195 30721]),'new4_sources','GDR','2026-TKDE-GDR-main');
addpath(genpath(source));addpath(fullfile(work,'compat'),'-begin');
addpath(fullfile(work,'sparse_adapter'),'-begin');
addpath(fullfile(work,'memory_safe_support','onehot_branch'),'-begin');
addpath(fullfile(work,'correct_onehot_compat'),'-begin');
maxNumCompThreads(2);global GDR_CAPTURE;GDR_CAPTURE=[];
assert(ismember(dataset,{'CIFAR10','CIFAR100'}) && all(ismember(seeds,1:5)));
assert(numel(unique(seeds))==numel(seeds));
gate=jsondecode(fileread(fullfile(work,'correct_onehot_combined_fixture.json')));assert(gate.all_passed);
if strcmp(dataset,'CIFAR10');file='cifar10.mat';else;file='cifar100.mat';end
input_path=fullfile(root,'baseline_fill_20260919','gap_data',file);
data=load(input_path,'Y');label=data.Y(:);clear data;
for seed=seeds
    out=fullfile(work,'correct_onehot_runs',dataset,sprintf('seed%d',seed));
    if ~exist(out,'dir');mkdir(out);end
    assert(~exist(fullfile(out,'native_result.mat'),'file'),'GDRCorrectOnehot:RefuseCifarOverwrite');
    diary(fullfile(out,'native_run.log'));
    config=struct('dataset',dataset,'seed',seed,'k',20,'h',12,'init_label','KM', ...
      'is_normal',1,'Maxiter',30,'cpu_only',true,'max_num_comp_threads',2, ...
      'source_default_h_claimed',false,'h_is_existing_explicit_resource_setting',true, ...
      'author_transformL_overridden',true,'independent_correct_onehot_semantic_patch',true, ...
      'initial_empty_cluster_guard_added',false,'cross_seed_graph_cache_used',false, ...
      'resource_adaptations','validated memory-bounded correct-onehot anchors, blocked sparse LearnB, no unused WB, file input', ...
      'original_CD_ALM_KM_stopping_retained',true,'uses_local_compatibility',true, ...
      'decision_basis','Class-encoding semantics; root selected branch before corrected-branch scores');
    fprintf('Start correct-onehot native %s seed%d at %s\n',dataset,seed,char(datetime('now')));
    rng(seed);
    [result,F,G,alpha,t,Obj,converge]=MvC_CGDR_sparse_onehot(input_path,[],20,12,'KM',1,30);
    [~,pred]=max(F,[],2);finite=all(isfinite(F(:))) && all(isfinite(alpha)) && all(isfinite(Obj));
    save(fullfile(out,'native_result.mat'),'result','pred','label','alpha','Obj', ...
      'converge','t','config','finite','G','-v7');
    fprintf('Complete correct-onehot native %s seed%d finite=%d seconds=%g outer=%d at %s\n', ...
      dataset,seed,finite,t,numel(Obj)-1,char(datetime('now')));
    fprintf('Metrics [ACC NMImax PUR Precision Recall Fscore ARI]: ');fprintf('%.15g ',result);fprintf('\n');diary off;
    assert(finite,'GDRCorrectOnehot:NonfiniteCifar');clear F G result alpha Obj pred;
end
fprintf('Requested correct-onehot CIFAR seeds complete.\n');
end
