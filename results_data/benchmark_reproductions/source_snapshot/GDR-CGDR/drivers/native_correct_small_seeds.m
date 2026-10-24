function native_correct_small_seeds()
work=fileparts(mfilename('fullpath'));root=fileparts(fileparts(fileparts(work)));
source=fullfile(root,'_teammate_all_datasets',char([48 51 95 35770 25991 22797 29616 20195 30721]),'new4_sources','GDR','2026-TKDE-GDR-main');
addpath(genpath(source));addpath(fullfile(work,'compat'),'-begin');
addpath(fullfile(work,'correct_onehot_compat'),'-begin');
maxNumCompThreads(2);global GDR_CAPTURE;GDR_CAPTURE=[];
expected=[1 0;0 1;1 0;0 1];assert(isequal(TransformL([1;2;1;2],2),expected));
assert(isequal(TransformL([2;2],2),[0 1;0 1]));
assert(isequal(TransformL([1;3;2],3),[1 0 0;0 0 1;0 1 0]));
% Exhaustive mathematical encoding gate for all binary vectors through N=8.
for n=1:8
    for code=0:2^n-1
        labels=bitget(code,1:n)'+1;H=TransformL(labels,2);
        assert(isequal(sum(H,2),ones(n,1)));
        [~,decoded]=max(H,[],2);assert(isequal(decoded,labels));
    end
end
gate=struct('all_passed',true,'binary_vectors_tested',sum(2.^(1:8)), ...
 'no_forced_nonempty_clusters',true,'no_RNG_calls',true,'author_source_modified',false);
fid=fopen(fullfile(work,'correct_onehot_semantic_gate.json'),'w');fprintf(fid,'%s',jsonencode(gate));fclose(fid);
datasets={'RGB-D','Mfeat'};
for di=1:numel(datasets)
    dataset=datasets{di};data=load(fullfile(root,'baseline_fill_20260919','gdr_data',[dataset '.mat']));
    X=data.X;label=data.Y(:);clear data;
    for seed=1:5
        out=fullfile(work,'correct_onehot_runs',dataset,sprintf('seed%d',seed));
        if ~exist(out,'dir');mkdir(out);end
        assert(~exist(fullfile(out,'native_result.mat'),'file'),'GDRCorrectOnehot:RefuseOverwrite');
        diary(fullfile(out,'native_run.log'));
        config=struct('dataset',dataset,'seed',seed,'k',20,'h',10,'init_label','KM', ...
          'is_normal',1,'Maxiter',30,'cpu_only',true,'max_num_comp_threads',2, ...
          'author_main_unchanged',true,'author_transformL_overridden',true, ...
          'independent_correct_onehot_semantic_patch',true,'uses_local_compatibility',true, ...
          'decision_basis','Encode each drawn StartInd class in its own column; branch chosen before corrected-branch scores', ...
          'initial_empty_cluster_guard_added',false);
        fprintf('Start correct-onehot native %s seed%d at %s\n',dataset,seed,char(datetime('now')));
        rng(seed);[result,F,G,alpha,t,Obj,converge]=MvC_CGDR(X,label,20,10,'KM',1,30);
        [~,pred]=max(F,[],2);finite=all(isfinite(F(:))) && all(isfinite(alpha)) && all(isfinite(Obj));
        save(fullfile(out,'native_result.mat'),'result','pred','label','alpha','Obj', ...
          'converge','t','config','finite','G','-v7');
        fprintf('Complete correct-onehot native %s seed%d finite=%d seconds=%g outer=%d at %s\n', ...
          dataset,seed,finite,t,numel(Obj)-1,char(datetime('now')));
        fprintf('Metrics [ACC NMImax PUR Precision Recall Fscore ARI]: ');fprintf('%.15g ',result);fprintf('\n');diary off;
        assert(finite,'GDRCorrectOnehot:Nonfinite');clear F G result alpha Obj pred;
    end
    clear X label;
end
fprintf('Correct-onehot RGB-D and Mfeat five-seed batches complete.\n');
end
