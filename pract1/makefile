DATA := data/
OUTPUT := output/

# Estilos que se utilizan para mostrar mensajes en el terminal
GREEN = \033[0;32m
CYAN = \033[0;36m
NO_COLOR = \033[0m
BOLD = \033[1m

all: mkdir ayuda_mp compila_mp ejecuta_mp compilar ayuda_perceptron ejecuta_perceptron ayuda_adaline ejecuta_adaline ejecuta_perceptron2 ejecuta_adaline2

mkdir:
	@mkdir -p $(OUTPUT)

# McCulloch-Pitts
ayuda_mp:
	@echo "${GREEN}Todos los comandos admite argumento ${CYAN}--f_out {nombre} ${GREEN}para indicar una salida, sino se imprime directamente por la pantalla.${NO_COLOR}\n"
	@echo "${GREEN}Para ejecutar McCulloch-Pitts, utilize el siguiente comando: ${CYAN}python3 McCulloch-Pitts.py --f_in data/entrada.txt ${GREEN}o${CYAN} make ejecuta_mp.${NO_COLOR}\n"
	@echo "${GREEN}En el caso de usar ${CYAN}make ejecuta_mp${GREEN}, se generará un fichero llamado mp_output en la carpeta output/.${NO_COLOR}\n"

compila_mp:
	@echo "Compilando todos los archivos relacionados con McCulloch-Pitts.\n"

ejecuta_mp:
	@echo "Ejecutando el script McCulloch-Pitts."
	python3 McCulloch-Pitts.py --f_in $(DATA)entrada.txt --f_out $(OUTPUT)mp.txt
	@echo ""

# Perceptron y Adaline Problema Real 1
compilar:
	@echo "${GREEN}Compilando todos los archivos relacionados con Perceptron y Adaline.${NO_COLOR}"

ayuda_perceptron:
	@echo "${GREEN}Todos los comandos admite argumento ${CYAN}--f_out {nombre} ${GREEN}para indicar una salida, sino se imprime directamente por la pantalla.${NO_COLOR}\n"
	@echo "${GREEN}Todos los comandos existe 3 modos de operacion ${CYAN}--modo1 {fichero} {porcion}; --modo2 {fichero}; --modo3 {train_file} {test_file}${NO_COLOR}\n"
	@echo "${GREEN}Un ejemplo del uso sería ${CYAN}python3 perceptron.py --modo2 $(DATA)and.txt --f_out $(OUTPUT)salida${NO_COLOR}\n"
	@echo "${GREEN}También se puede indicar los hiperparámetros mediante siguientes Flags: ${CYAN}--umbral {num} --alpha {num} --epoca{num} ${NO_COLOR}\n"
	@echo "${GREEN}En el caso de usar ${CYAN}make ejecuta_perceptron${GREEN}, se generarán los fichero en la carpeta output/.${NO_COLOR}\n"

ejecuta_perceptron:
	@echo "Ejecutando el script Perceptron."
	python3 perceptron.py --modo1 $(DATA)problema_real1.txt 0.25 --f_out $(OUTPUT)problema_real1_per.txt --umbral 0.01 --alpha 0.125 --epoca 800
	@echo ""

ayuda_adaline:
	@echo "${GREEN}Todos los comandos admite argumento ${CYAN}--f_out {nombre} ${GREEN}para indicar una salida, sino se imprime directamente por la pantalla.${NO_COLOR}\n"
	@echo "${GREEN}Todos los comandos existe 3 modos de operacion ${CYAN}--modo1 {fichero} {porcion}; --modo2 {fichero}; --modo3 {train_file} {test_file}${NO_COLOR}\n"
	@echo "${GREEN}Un ejemplo del uso sería ${CYAN}python3 adaline.py --modo2 $(DATA)and.txt --f_out $(OUTPUT)salida${NO_COLOR}\n"
	@echo "${GREEN}También se puede indicar los hiperparámetros mediante siguientes Flags: ${CYAN}--umbral {num} --alpha {num} --epoca{num} --torelancia {num} ${NO_COLOR}\n"
	@echo "${GREEN}En el caso de usar ${CYAN}make ejecuta_perceptron${GREEN}, se generarán los fichero en la carpeta output/.${NO_COLOR}\n"

ejecuta_adaline:
	@echo "Ejecutando el script Adaline."
	python3 adaline.py --modo1 $(DATA)problema_real1.txt 0.25 --f_out $(OUTPUT)problema_real1_ada.txt --umbral 0.1 --alpha 0.025 --torelancia 0.1
	@echo ""

# Perceptron y Adaline Problema Real 2
ejecuta_perceptron2:
	@echo "Ejecutando el script Perceptron. Problema real 2."
	python3 perceptron.py --modo3 $(DATA)problema_real2.txt $(DATA)problema_real2_no_etiquetados.txt --f_out $(OUTPUT)problema_real2_per.txt --umbral 0.1 --alpha 0.025 --epoca 400
	@echo ""

ejecuta_adaline2:
	@echo "Ejecutando el script Adaline. Problema real 2."
	python3 adaline.py --modo3 $(DATA)problema_real2.txt $(DATA)problema_real2_no_etiquetados.txt --f_out $(OUTPUT)problema_real2_ada.txt --umbral 0.1 --alpha 0.025 --torelancia 0.1
	@echo ""

clean:
	rm -df salida.txt $(OUTPUT)