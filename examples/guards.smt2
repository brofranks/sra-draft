; One input keeps the prefix choice correlated with both advertisements.
(set-logic QF_BV)
(declare-const x (_ BitVec 3))
(define-fun prefix () Bool (= ((_ extract 2 2) x) #b1))
(define-fun a () Bool (= ((_ extract 1 1) x) #b1))
(define-fun b () Bool (= ((_ extract 0 0) x) #b1))
(define-fun permitted () Bool (ite prefix b a))
