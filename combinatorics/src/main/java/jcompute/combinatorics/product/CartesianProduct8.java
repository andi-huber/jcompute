/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.combinatorics.product;

import java.math.BigInteger;
import java.util.Objects;
import java.util.Optional;
import java.util.function.IntFunction;
import java.util.stream.Gatherer;
import java.util.stream.Stream;

import jcompute.core.util.function.MultiIntConsumer;
import jcompute.core.util.function.MultiIntPredicate;
import jcompute.core.util.function.PrefixedMultiIntConsumer;

public record CartesianProduct8(int n0, int n1, int n2, int n3, int n4, int n5, int n6, int n7) implements CartesianProduct {

    @Override public int indexCount() { return 8; }
    @Override public BigInteger cardinality() {
        return BigInteger.valueOf(n0)
            .multiply(BigInteger.valueOf(n1))
            .multiply(BigInteger.valueOf(n2))
            .multiply(BigInteger.valueOf(n3))
            .multiply(BigInteger.valueOf(n4))
            .multiply(BigInteger.valueOf(n5))
            .multiply(BigInteger.valueOf(n6))
            .multiply(BigInteger.valueOf(n7));
    }

    @Override
    public void reportIndexRanges(final MultiIntConsumer intConsumer) {
        intConsumer.accept(n0, n1, n2, n3, n4, n5, n6, n7);
    }

    @Override
    public void forEach(final Visiting visiting, final MultiIntConsumer intConsumer) {
        visiting.range(n0).forEach(i->{
            for(int j=0; j<n1; ++j){
                for(int k=0; k<n2; ++k){
                    for(int l=0; l<n3; ++l){
                        for(int m=0; m<n4; ++m){
                            for(int n=0; n<n5; ++n){
                                for(int o=0; o<n6; ++o){
                                    for(int p=0; p<n7; ++p){
                                        intConsumer.accept(i, j, k, l, m, n, o, p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    @Override
    public Stream<int[]> stream(final Visiting visiting) {
        return visiting.range(n0)
            .mapToObj(Integer::valueOf)
            .gather(Gatherer.of(new Integrators.Integrator8(n1, n2, n3, n4, n5, n6, n7)));
    }

    @Override
    public <T> Stream<T> streamCollectors(final IntFunction<T> collectorFactory, final PrefixedMultiIntConsumer<T> prefixedIntConsumer) {
        return Visiting.PARALLEL.range(n0).mapToObj(i->{
            T t = collectorFactory.apply(i);
            for(int j=0; j<n1; ++j){
                for(int k=0; k<n2; ++k){
                    for(int l=0; l<n3; ++l){
                        for(int m=0; m<n4; ++m){
                            for(int n=0; n<n5; ++n){
                                for(int o=0; o<n6; ++o){
                                    for(int p=0; p<n7; ++p){
                                        prefixedIntConsumer.accept(t, i, j, k, l, m, n, o, p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return t;
        });
    }

    @Override
    public Optional<int[]> findAny(final MultiIntPredicate intPredicate) {
        return Visiting.PARALLEL.range(n0)
            .mapToObj(i->{
                for(int j=0; j<n1; ++j){
                    for(int k=0; k<n2; ++k){
                        for(int l=0; l<n3; ++l){
                            for(int m=0; m<n4; ++m){
                                for(int n=0; n<n5; ++n){
                                    for(int o=0; o<n6; ++o){
                                        for(int p=0; p<n7; ++p){
                                            if(intPredicate.test(i, j, k, l, m, n, o, p)) {
                                                return new int[] {i, j, k, l, m, n, o, p};
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return (int[]) null;
            })
            .filter(Objects::nonNull)
            .findAny();
    }

}
