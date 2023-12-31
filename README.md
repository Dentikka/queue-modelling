# Возникновение пробок и явление фазового перехода в системах массового обслуживания

Рассматривается задача о росте очереди. Пример такой задачи - касса в магазине

## Постановка задачи
Люди подходят к кассе с интенсивностью $\varphi(t)$ - монотонно неубывающей со временем; кассир обслуживает людей с постоянной интенсивностью $\lambda$. Ставится вопрос: насколько быстро будет меняться среднее число людей в очереди? Поняв это, можно, например, предсказывать через сколько времени придётся открыть вторую кассу и решать другие прикладные задачи.

## Стационарный случай
Рассатривается стационарный марковский процесс. Для него выводится динамика роста очереди, в которой наблюдается фазовый переход

## Численное решение
Для общего случая решается система дифференциальных уравнений методом Рунге-Кутты 4 порядка

## Моделирование
Моделирование происходит следующим образом. Рассматриваемый промежуток времени разбивается на отрезки длины $dt$. 
На каждом отрезке $[t, t + dt]$ интенсивность прихода новых людей в очередь считается постоянной. 
Вычисляется вероятность появления нового человека в очередди на этом отрезке как вероятность этого отрезка в экспоненциальном распределении с интенсивностью $\varphi(t)$.
Семплируется бернуллиевская случайная величина с такой вероятностью. Аналогично семплируется уход человека из очереди.
В итоге строется траектория зависимости длины очереди в разные моменты времени с шагом $dt$.
Строится несколько таких траекторий, затем результаты усредняются.
