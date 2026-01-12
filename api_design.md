FAST API EM PYTHON DO ASSET MANAGEMENT

Function get_user_assets GET:
- params: integer clientid
- returns: JSON {"principal": 1000 BRD de fato, "lucro": 100 BRL (BRD de direito), se for feito o staking, 100 BRDs vao ser criados e devolvidos ao usuario}
Retorna a quantidade de capital que o usuario tem na plataforma em reais

Caso de uso
Usuario A - criou 1000 reais e fez staking em 1000 reais
Usuario B - criou 500 reais e nao fez staking em nada

Em 1 ano, assumindo 10% ao ano
Caixa total 1500 principal + 150 de rendimento
1000 + 100 - pertencem ao Usuario A
500 + 0 pertencem ao Usuario B
0 + 50 reais pertencem ao BRD

Exemplo: get clientid_usuario A -> {"principal": 1000, "lucro": 100}

Database:
clientid,action,value,timestamp,expirationdate
- action -> create,redeem,staking,unstaking


Function POST create_brd
sub function cria BRD's DO PRINCIPAL
- params: integer clientid, float value (BRD)
- returns: HTTP response 200, 404 (timestamp in text)
- execution: comprar LFT equivalente ao valor depositado, salva o valor de execucao parcial no banco de asset management


Function POST staking_brd
- params: integer clientid, float value (BRD)
- returns: HTTP response
- execution: insert on database the date and value of staking


Function POST unstaking_brd
sub function cria BRD's DO LUCRO
- param: integer clientid, float value (BRD)
- returns: HTTP response
- execution: faz um um insert no banco com a acao, os retornos de capital passama agora a ser da BRD -> trigger processo de mint de BRD's associados aos lucros e perdas do investimento de staking


Function POST redeem_brd
sub function destroi BRD's DO PRINCIPAL + LUCRO
- params: integer clientid, float value (BRD)
- returns: HTTP response 200, 404 (timestamp in text) -> deposita em BRL na conta do cliente
- exection: Vende LFT equivalente ao valor redeemed, envia o capital para o conta do banco genial, insere no banco de dados a acao e a data, insere no banco de asset management as parciais de execucao
