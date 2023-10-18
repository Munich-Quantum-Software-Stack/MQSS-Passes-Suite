; type definitions

%Result = type opaque
%Qubit = type opaque

; global constants (labels for output recording)

@0 = internal constant [3 x i8] c"r1\00"
@1 = internal constant [3 x i8] c"r2\00"

@d0 = internal constant double 0.0

; entry point definition

define i64 @Entry_Point_Name() #0 {
entry:
  %0 = sdiv i32 1, 0
  %zero = load double, double* @d0
  ; calls to initialize the execution environment
  call void @__quantum__rt__initialize(i8* null)
  ; calls to QIS functions that are not irreversible
  call void @__quantum__qis__U3__body(double %zero, double %zero, double %zero, %Qubit* null)
  call void @__quantum__qis__rz__body(double %zero, %Qubit* null)
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__x__body(%Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__x__body(%Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__rx__body(double %zero, %Qubit* inttoptr (i64 1 to %Qubit*))
  ; calls to QIS functions that are irreversible
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* writeonly null)
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* writeonly inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__id__body(%Qubit* null)
  ; calls to record the program output
  call void @__quantum__rt__tuple_record_output(i64 2, i8* null)
  call void @__quantum__rt__result_record_output(%Result* null, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  br i1 true, label %true_block, label %false_block

true_block:                                       ; preds = %entry
  br label %exit_block

false_block:                                      ; preds = %entry
  br label %exit_block

exit_block:                                       ; preds = %false_block, %true_block
  ret i64 0
}

define i64 @NonEntry_Point_Name() #2 {
entry:
  ret i64 0
}

; declarations of QIS functions

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__U3__body(double, double, double, %Qubit*)

declare void @__quantum__qis__rz__body(double, %Qubit*)

declare void @__quantum__qis__rx__body(double, %Qubit*)

declare void @__quantum__qis__x__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare void @__quantum__qis__cz__body(%Qubit*, %Qubit*)

declare void @__quantum__qis__id__body(%Qubit*)

declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #1

; declarations of runtime functions for initialization and output recording

declare void @__quantum__rt__initialize(i8*)

declare void @__quantum__rt__tuple_record_output(i64, i8*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

; attributes

attributes #0 = { "entry_point" "qir_profiles"="base_profile" "output_labeling_schema"="schema_id" "num_required_qubits"="2" "num_required_results"="2" }

attributes #1 = { "irreversible" }

attributes #2 = { "qir_profiles"="base_profile" }

; module flags

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}

