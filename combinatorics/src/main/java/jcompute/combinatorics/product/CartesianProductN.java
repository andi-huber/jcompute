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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntFunction;
import java.util.stream.Gatherer;
import java.util.stream.Gatherer.Downstream;
import java.util.stream.Gatherer.Integrator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import jcompute.core.util.function.MultiIntConsumer;
import jcompute.core.util.function.MultiIntPredicate;
import jcompute.core.util.function.PrefixedMultiIntConsumer;

public record CartesianProductN(int ...dim) implements CartesianProduct {

    @Override public int indexCount() { return dim.length; }
    @Override public BigInteger cardinality() {
        return IntStream.of(dim)
            .mapToObj(BigInteger::valueOf)
            .reduce(BigInteger.ONE, BigInteger::multiply);
    }

    @Override
    public void reportIndexRanges(final MultiIntConsumer intConsumer) {
        intConsumer.accept(dim);
    }

    @Override
    public void forEach(final Visiting visiting, final MultiIntConsumer intConsumer) {
        visiting.range(dim[0]).forEach(i->{
            var v = new int[dim.length];
            v[0] = i;
            new RecursiveVisitor(dim, v, intConsumer).recur(1);
        });
    }

    @Override
    public Stream<int[]> stream(final Visiting visiting) {
        return visiting.range(dim[0])
            .mapToObj(Integer::valueOf)
            .gather(Gatherer.of(new IntegratorN(dim)));
    }

    @Override
    public <T> Stream<T> streamCollectors(final IntFunction<T> collectorFactory, final PrefixedMultiIntConsumer<T> prefixedIntConsumer) {
        return Visiting.PARALLEL.range(dim[0]).mapToObj(i->{
            var t = collectorFactory.apply(i);
            var v = new int[dim.length];
            v[0] = i;
            new RecursiveCollector<>(dim, v, t, prefixedIntConsumer).recur(1);
            return t;
        });
    }

    @Override
    public Optional<int[]> findAny(final MultiIntPredicate intPredicate) {
        final AtomicReference<int[]> result = new AtomicReference<>();
        return Visiting.PARALLEL.range(dim[0])
            .mapToObj(i->{
                if(result.get()!=null) return null;
                var v = new int[dim.length];
                v[0] = i;
                new RecursiveFinder(dim, v, intPredicate, result).recur(1);
                return result.get();
            })
            .filter(Objects::nonNull)
            .findAny();
    }

    // -- HELPER

    private record IntegratorN(int ...dim)
    implements Integrator<Void, Integer, int[]> {
        @Override
        public boolean integrate(final Void state, final Integer i, final Downstream<? super int[]> downstream) {
            final MultiIntPredicate mip = downstream::push;
            final AtomicBoolean stop = new AtomicBoolean();
            var v = new int[dim.length];
            v[0] = i;
            new RecursiveWhile(dim, v, mip::test, stop).recur(1);
            return !stop.get();
        }
    }

    private record RecursiveVisitor(int[] dim, int[] v, MultiIntConsumer intConsumer) {
        void recur(final int dimIndex){
            if(dimIndex == v.length) {
                intConsumer.accept(v);
                return;
            }
            for(int i=0; i<dim[dimIndex]; ++i){
                v[dimIndex] = i;
                recur(dimIndex + 1);
            }
        }
    }

    private record RecursiveCollector<T>(int[] dim, int[] v, T t, PrefixedMultiIntConsumer<T> prefixedIntConsumer) {
        void recur(final int dimIndex){
            if(dimIndex == v.length) {
                prefixedIntConsumer.accept(t, v);
                return;
            }
            for(int i=0; i<dim[dimIndex]; ++i){
                v[dimIndex] = i;
                recur(dimIndex + 1);
            }
        }
    }

    private record RecursiveFinder(int[] dim, int[] v, MultiIntPredicate intPredicate, AtomicReference<int[]> result) {
        void recur(final int dimIndex){
            if(dimIndex == v.length) {
                if(intPredicate.test(v)) {
                    result.set(v.clone());
                }
                return;
            }
            for(int i=0; i<dim[dimIndex]; ++i){
                v[dimIndex] = i;
                recur(dimIndex + 1);
            }
        }
    }

    private record RecursiveWhile(int[] dim, int[] v, MultiIntPredicate condition, AtomicBoolean stop) {
        void recur(final int dimIndex){
            if(dimIndex == v.length) {
                if(!condition.test(v)) {
                    stop.set(true);
                }
                return;
            }
            for(int i=0; i<dim[dimIndex]; ++i){
                v[dimIndex] = i;
                recur(dimIndex + 1);
            }
        }
    }


}
